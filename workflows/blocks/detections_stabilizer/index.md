
# Detections Stabilizer



??? "Class: `StabilizeTrackedDetectionsBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/transformations/stabilize_detections/v1.py">inference.core.workflows.core_steps.transformations.stabilize_detections.v1.StabilizeTrackedDetectionsBlockV1</a>
    



Apply smoothing algorithms to reduce noise and flickering in tracked detections across video frames by using Kalman filtering to predict object velocities, exponential moving average to smooth bounding box positions, and gap filling to restore temporarily missing detections for improved tracking stability and smoother visualization workflows.

## How This Block Works

This block stabilizes tracked detections by reducing jitter, smoothing positions, and filling gaps when objects temporarily disappear from detection. The block:

1. Receives tracked detection predictions with unique tracker IDs and an image with embedded video metadata
2. Extracts video metadata from the image:
   - Accesses video_metadata to get video_identifier
   - Uses video_identifier to maintain separate stabilization state for different videos
3. Validates that detections have tracker IDs (required for tracking object movement across frames)
4. Initializes or retrieves stabilization state for the video:
   - Maintains a cache of last known detections for each tracker_id per video
   - Creates or retrieves a Kalman filter for velocity prediction per video
   - Stores separate state for each video using video_identifier
5. Measures object velocities for existing tracks:
   - Calculates velocity by comparing current frame bounding box centers to previous frame centers
   - Computes displacement (change in position) for objects present in both current and previous frames
   - Velocity measurements are used to update the Kalman filter
6. Updates Kalman filter with velocity measurements:
   - Uses Kalman filtering to predict smoothed velocities based on historical measurements
   - Maintains a sliding window of velocity measurements (controlled by smoothing_window_size)
   - Applies exponential moving average within the Kalman filter to smooth velocity estimates
   - Filters out noise from detection inaccuracies and frame-to-frame variations
7. Smooths bounding boxes for objects present in current frame:
   - Applies exponential moving average smoothing to bounding box coordinates
   - Combines previous frame position with current frame position using bbox_smoothing_coefficient
   - Formula: smoothed_bbox = alpha * current_bbox + (1 - alpha) * previous_bbox
   - Reduces jitter and flickering from detection variations
8. Predicts positions for missing detections:
   - Uses Kalman filter predicted velocities to estimate positions of objects that disappeared
   - Applies predicted velocity to last known bounding box position
   - Fills gaps by restoring detections that were temporarily missing from current frame
   - Smooths predicted positions using exponential moving average
9. Manages tracking state:
   - Updates cache with current frame detections for next frame calculations
   - Removes tracking entries for objects that have been missing longer than smoothing_window_size frames
   - Maintains separate state per video_identifier
10. Merges and returns stabilized detections:
    - Combines smoothed detections (from current frame) and predicted detections (for missing objects)
    - Outputs stabilized detection objects with reduced noise and filled gaps
    - All detections maintain their tracker IDs for consistent tracking

The block uses two complementary smoothing techniques: **Kalman filtering** for velocity prediction (estimating how fast objects are moving) and **exponential moving average** for position smoothing (reducing bounding box jitter). The Kalman filter maintains a history of velocity measurements and uses statistical estimation to predict future velocities while filtering out noise. The exponential moving average smooths bounding box coordinates by blending current and previous positions. Gap filling uses predicted velocities to restore detections that temporarily disappear, helping maintain track continuity. Note: This block may produce short-lived bounding boxes for unstable trackers, as it attempts to fill gaps even when objects are inconsistently detected.

## Common Use Cases

- **Video Visualization**: Reduce flickering and jitter in video annotations for smoother visualizations (e.g., smooth bounding box movements, reduce annotation noise, improve video visualization quality), enabling stable video visualization workflows
- **Tracking Stability**: Improve tracking stability when detections are noisy or inconsistent (e.g., stabilize noisy detections, reduce tracking jitter, improve tracking continuity), enabling stable tracking workflows
- **Temporary Occlusion Handling**: Fill gaps when objects are temporarily occluded or missing from detections (e.g., maintain tracks during brief occlusions, fill detection gaps, preserve tracking continuity), enabling occlusion handling workflows
- **Real-Time Monitoring**: Improve visual quality in real-time monitoring applications (e.g., smooth live video annotations, reduce flickering in monitoring displays, improve real-time visualization), enabling stable real-time monitoring workflows
- **Analytics Accuracy**: Reduce noise in analytics calculations that depend on stable detection positions (e.g., improve position-based analytics, reduce noise in measurements, stabilize movement calculations), enabling accurate analytics workflows
- **Quality Control**: Improve detection quality for downstream processing (e.g., smooth detections before analysis, reduce noise for better processing, stabilize inputs for other blocks), enabling quality improvement workflows

## Connecting to Other Blocks

This block receives tracked detections and an image, and produces stabilized tracked_detections:

- **After Byte Tracker blocks** to stabilize tracked detections (e.g., smooth tracked object positions, reduce tracking jitter, fill tracking gaps), enabling tracking-stabilization workflows
- **After object detection or instance segmentation blocks** with tracking enabled to stabilize detections (e.g., smooth detection positions, reduce detection noise, improve tracking stability), enabling detection-stabilization workflows
- **Before visualization blocks** to display stabilized detections (e.g., visualize smooth bounding boxes, display stable annotations, show gap-filled detections), enabling stable visualization workflows
- **Before analytics blocks** to provide stable inputs for analysis (e.g., analyze stabilized positions, process smooth movement data, work with gap-filled detections), enabling stable analytics workflows
- **Before velocity or path analysis blocks** to improve measurement accuracy (e.g., calculate velocities from stable positions, analyze paths from smooth trajectories, measure from gap-filled detections), enabling accurate measurement workflows
- **In video processing pipelines** where detection stability is required for downstream processing (e.g., stabilize detections in processing chains, improve quality for analysis, reduce noise in pipelines), enabling stable video processing workflows

## Requirements

This block requires tracked detections with tracker_id information (detections must come from a tracking block like Byte Tracker). The image's video_metadata should include video_identifier to maintain separate stabilization state for different videos. The block maintains persistent stabilization state across frames for each video, so it should be used in video workflows where frames are processed sequentially. For optimal stabilization, detections should be provided consistently across frames with valid tracker IDs. The smoothing_window_size controls how many historical velocity measurements are used for Kalman filtering and how long missing detections are retained. The bbox_smoothing_coefficient (0-1) controls the balance between current and previous positions - lower values provide more smoothing but slower response to changes, higher values provide less smoothing but faster response. Note: This block may produce short-lived bounding boxes for unstable trackers as it attempts to fill gaps even when objects are inconsistently detected.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/stabilize_detections@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `smoothing_window_size` | `int` | Size of the sliding window for velocity smoothing in Kalman filter, controlling how many historical velocity measurements are used. Also determines how long missing detections are retained before removal. Larger values provide more smoothing but slower adaptation to changes. Smaller values provide less smoothing but faster adaptation. Detections missing for longer than this number of frames are removed from tracking state. Typical range: 3-10 frames.. | ✅ |
| `bbox_smoothing_coefficient` | `float` | Exponential moving average coefficient (alpha) for bounding box position smoothing, range 0.0-1.0. Controls the blend between current and previous bounding box positions: smoothed_bbox = alpha * current + (1-alpha) * previous. Lower values (closer to 0) provide more smoothing - slower response to changes, less jitter. Higher values (closer to 1) provide less smoothing - faster response to changes, more jitter. Default 0.2 balances smoothness and responsiveness. Typical range: 0.1-0.5.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-alert-circle-outline:{ style="color: #f57c00" } `soft` — runtime `hosted_serverless`, `dedicated_deployment`; execution `remote`; input `video`
:   Block keeps per-video state in process memory (keyed by video_metadata.video_identifier). With remote step execution on stateless or multi-replica HTTP runtimes, successive requests may be served by different worker processes, so the state resets between calls and the output is meaningless for tracking / counting / aggregation. Use local step execution in an InferencePipeline for stable cross-frame results.

:material-alert-circle-outline:{ style="color: #f57c00" } `soft` — input `image`
:   Block depends on temporal context from video or repeated-frame workflows. With a still image/photo, there is no meaningful history to track, compare, aggregate, or visualize, so the block provides little or no benefit.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Detections Stabilizer` in version `v1`.

    - inputs: [`Detections Filter`](detections_filter.md), [`Path Deviation`](path_deviation.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`SAM 3`](sam3.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Image Stack`](image_stack.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Triangle Visualization`](triangle_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`EasyOCR`](easy_ocr.md), [`Dynamic Zone`](dynamic_zone.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Camera Focus`](camera_focus.md), [`Image Slicer`](image_slicer.md), [`Track Class Lock`](track_class_lock.md), [`YOLO-World Model`](yolo_world_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Overlap Filter`](overlap_filter.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Stitch Images`](stitch_images.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Time in Zone`](timein_zone.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Dynamic Crop`](dynamic_crop.md), [`Perspective Correction`](perspective_correction.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Line Counter`](line_counter.md), [`Polygon Visualization`](polygon_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`VLM As Detector`](vlm_as_detector.md), [`OCR Model`](ocr_model.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Circle Visualization`](circle_visualization.md), [`Line Counter`](line_counter.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Grid Visualization`](grid_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Distance Measurement`](distance_measurement.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Time in Zone`](timein_zone.md), [`Byte Tracker`](byte_tracker.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Corner Visualization`](corner_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Image Blur`](image_blur.md), [`Label Visualization`](label_visualization.md), [`Template Matching`](template_matching.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Seg Preview`](seg_preview.md), [`Path Deviation`](path_deviation.md), [`Object Detection Model`](object_detection_model.md), [`Pixel Color Count`](pixel_color_count.md), [`Velocity`](velocity.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Image Preprocessing`](image_preprocessing.md), [`Trace Visualization`](trace_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Color Visualization`](color_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Detections Consensus`](detections_consensus.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Calibration`](camera_calibration.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Dot Visualization`](dot_visualization.md), [`Motion Detection`](motion_detection.md), [`Halo Visualization`](halo_visualization.md), [`Detection Offset`](detection_offset.md), [`SAM 3`](sam3.md), [`Google Vision OCR`](google_vision_ocr.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Image Contours`](image_contours.md), [`Camera Focus`](camera_focus.md), [`Halo Visualization`](halo_visualization.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Moondream2`](moondream2.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Object Detection Model`](object_detection_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Relative Static Crop`](relative_static_crop.md), [`Detections Combine`](detections_combine.md), [`Contrast Equalization`](contrast_equalization.md), [`Identify Changes`](identify_changes.md), [`Image Slicer`](image_slicer.md), [`Detections Merge`](detections_merge.md), [`Mask Area Measurement`](mask_area_measurement.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Byte Tracker`](byte_tracker.md), [`SAM 3`](sam3.md), [`Detections Stitch`](detections_stitch.md), [`Text Display`](text_display.md), [`Depth Estimation`](depth_estimation.md), [`Image Threshold`](image_threshold.md), [`Icon Visualization`](icon_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Time in Zone`](timein_zone.md), [`SIFT Comparison`](sift_comparison.md), [`SIFT`](sift.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Background Subtraction`](background_subtraction.md), [`Identify Outliers`](identify_outliers.md), [`Object Detection Model`](object_detection_model.md), [`Background Color Visualization`](background_color_visualization.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Mask Visualization`](mask_visualization.md)
    - outputs: [`Path Deviation`](path_deviation.md), [`Detections Filter`](detections_filter.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Triangle Visualization`](triangle_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Crop Visualization`](crop_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Overlap Filter`](overlap_filter.md), [`Track Class Lock`](track_class_lock.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Time in Zone`](timein_zone.md), [`Polygon Visualization`](polygon_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Dynamic Crop`](dynamic_crop.md), [`Event Writer`](event_writer.md), [`Perspective Correction`](perspective_correction.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Line Counter`](line_counter.md), [`Polygon Visualization`](polygon_visualization.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Circle Visualization`](circle_visualization.md), [`Line Counter`](line_counter.md), [`Distance Measurement`](distance_measurement.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Time in Zone`](timein_zone.md), [`Byte Tracker`](byte_tracker.md), [`Florence-2 Model`](florence2_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Overlap Analysis`](overlap_analysis.md), [`SORT Tracker`](sort_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Corner Visualization`](corner_visualization.md), [`Label Visualization`](label_visualization.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Path Deviation`](path_deviation.md), [`Velocity`](velocity.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Trace Visualization`](trace_visualization.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Detections Consensus`](detections_consensus.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Dot Visualization`](dot_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Detection Offset`](detection_offset.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Camera Focus`](camera_focus.md), [`Halo Visualization`](halo_visualization.md), [`Detections Combine`](detections_combine.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Detections Merge`](detections_merge.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Byte Tracker`](byte_tracker.md), [`Detections Stitch`](detections_stitch.md), [`Size Measurement`](size_measurement.md), [`Icon Visualization`](icon_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Time in Zone`](timein_zone.md), [`GeoTag Detection`](geo_tag_detection.md), [`Background Color Visualization`](background_color_visualization.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Mask Visualization`](mask_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Detections Stabilizer` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Image with embedded video metadata. The video_metadata contains video_identifier to maintain separate stabilization state for different videos. Required for persistent state management across frames..
        - `detections` (*Union[[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md)]*): Tracked object detection or instance segmentation predictions. Must include tracker_id information from a tracking block. The block applies Kalman filtering for velocity prediction, exponential moving average for position smoothing, and gap filling for missing detections. Output detections are stabilized with reduced noise and jitter..
        - `smoothing_window_size` (*[`integer`](../kinds/integer.md)*): Size of the sliding window for velocity smoothing in Kalman filter, controlling how many historical velocity measurements are used. Also determines how long missing detections are retained before removal. Larger values provide more smoothing but slower adaptation to changes. Smaller values provide less smoothing but faster adaptation. Detections missing for longer than this number of frames are removed from tracking state. Typical range: 3-10 frames..
        - `bbox_smoothing_coefficient` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Exponential moving average coefficient (alpha) for bounding box position smoothing, range 0.0-1.0. Controls the blend between current and previous bounding box positions: smoothed_bbox = alpha * current + (1-alpha) * previous. Lower values (closer to 0) provide more smoothing - slower response to changes, less jitter. Higher values (closer to 1) provide less smoothing - faster response to changes, more jitter. Default 0.2 balances smoothness and responsiveness. Typical range: 0.1-0.5..

    - output
    
        - `tracked_detections` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Prediction with detected bounding boxes in form of sv.Detections(...) object if `object_detection_prediction` or Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object if `instance_segmentation_prediction`.



??? tip "Example JSON definition of step `Detections Stabilizer` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/stabilize_detections@v1",
	    "image": "<block_does_not_provide_example>",
	    "detections": "$steps.object_detection_model.predictions",
	    "smoothing_window_size": 3,
	    "bbox_smoothing_coefficient": 0.2
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    



# Velocity



??? "Class: `VelocityBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/analytics/velocity/v1.py">inference.core.workflows.core_steps.analytics.velocity.v1.VelocityBlockV1</a>
    



Calculate the velocity and speed of tracked objects across video frames by measuring displacement of object centers over time, applying exponential moving average smoothing to reduce noise, and converting measurements from pixels per second to meters per second for traffic speed monitoring, movement analysis, behavior tracking, and performance measurement workflows.

## How This Block Works

This block measures how fast objects are moving by tracking their positions across video frames. The block:

1. Receives tracked detection predictions with unique tracker IDs and an image with embedded video metadata
2. Extracts video metadata from the image:
   - Accesses video_metadata to get frame timestamps or frame numbers and frame rate (fps)
   - Extracts video_identifier to maintain separate tracking state for different videos
   - Determines current timestamp using frame_number/fps for video files or frame_timestamp for streams
3. Validates that detections have tracker IDs (required for tracking object movement across frames)
4. Calculates object center positions:
   - Computes the center point (x, y) of each bounding box in the current frame
   - Uses bounding box coordinates to find geometric centers
5. Retrieves or initializes tracking state:
   - Maintains previous positions and timestamps for each tracker_id per video
   - Stores smoothed velocity history for each tracker_id per video
   - Creates new tracking entries for objects appearing for the first time
6. Calculates velocity and speed for each tracked object:
   - **For objects with previous positions**: Computes displacement (change in position) and time delta (change in time) between current and previous frames
   - **Velocity**: Calculates velocity vector as displacement divided by time delta (pixels per second)
   - **Speed**: Computes speed as the magnitude (length) of the velocity vector (total pixels per second regardless of direction)
   - **For new objects**: Sets velocity and speed to zero (no movement data available yet)
7. Applies exponential moving average smoothing:
   - Smooths velocity measurements using exponential moving average with configurable smoothing factor (alpha)
   - Reduces noise and jitter in velocity calculations from detection variations
   - Lower alpha values provide more smoothing (slower response to changes), higher alpha values provide less smoothing (faster response to changes)
   - Calculates smoothed velocity and smoothed speed for each object
8. Converts units to meters per second:
   - Divides pixel-based velocities and speeds by pixels_per_meter conversion factor
   - Converts all measurements (velocity, speed, smoothed_velocity, smoothed_speed) to real-world units
   - Enables comparison with real-world speed measurements (e.g., km/h, mph)
9. Stores velocity data in detection metadata:
   - Adds four velocity metrics to each detection: velocity (m/s), speed (m/s), smoothed_velocity (m/s), smoothed_speed (m/s)
   - Velocity is a 2D vector [vx, vy] representing direction and magnitude of movement
   - Speed is a scalar value representing total speed regardless of direction
   - All measurements are stored in detections.data for downstream use
10. Updates tracking state for next frame:
    - Saves current positions and timestamps for all tracked objects
    - Stores smoothed velocities for next frame's smoothing calculations
11. Returns detections enhanced with velocity information:
    - Outputs the same detection objects with added velocity metadata
    - Each detection now includes velocity and speed data in its metadata

Velocity is calculated based on the displacement of object centers (bounding box centers) over time. The block maintains separate tracking state for each video, allowing velocity calculation across multiple video streams. Due to perspective distortion and camera positioning, calculated velocity may vary depending on where objects appear in the frame - objects closer to the camera or at different depths will have different pixel-per-second values for the same real-world speed. The smoothing helps reduce noise from detection inaccuracies and frame-to-frame variations.

## Common Use Cases

- **Traffic Speed Monitoring**: Measure vehicle speeds on roads and highways (e.g., monitor traffic speeds, detect speeding violations, analyze traffic flow rates), enabling traffic enforcement and analysis workflows
- **Sports Performance Analysis**: Track athlete movement and speed during sports activities (e.g., measure player speeds, analyze sprint performance, track movement patterns), enabling sports analytics workflows
- **Security and Surveillance**: Monitor movement speed of people or objects in security scenarios (e.g., detect running or suspicious rapid movement, monitor crowd flow speeds, track object movement rates), enabling security monitoring workflows
- **Retail Analytics**: Analyze customer movement patterns and walking speeds in retail spaces (e.g., measure customer flow rates, analyze shopping behavior patterns, track movement efficiency), enabling retail behavior analysis workflows
- **Wildlife Behavior Studies**: Track animal movement speeds and patterns in natural habitats (e.g., measure animal speeds, analyze migration patterns, study movement behavior), enabling wildlife research workflows
- **Industrial Monitoring**: Monitor speeds of vehicles, equipment, or products in industrial settings (e.g., track conveyor speeds, measure vehicle speeds in facilities, monitor production line movement rates), enabling industrial automation workflows

## Connecting to Other Blocks

This block receives tracked detections and an image with embedded video metadata, and produces detections enhanced with velocity metadata:

- **After Byte Tracker blocks** to calculate velocity for tracked objects (e.g., measure speeds of tracked vehicles, analyze tracked person movement, monitor tracked object velocities), enabling tracking-to-velocity workflows
- **After object detection or instance segmentation blocks** with tracking enabled to measure movement speeds (e.g., calculate vehicle speeds, track person movement rates, monitor object velocities), enabling detection-to-velocity workflows
- **Before visualization blocks** to display velocity information (e.g., visualize speed overlays, display velocity vectors, show movement speed annotations), enabling velocity visualization workflows
- **Before logic blocks** like Continue If to make decisions based on speed thresholds (e.g., continue if speed exceeds limit, filter based on velocity ranges, trigger actions on speed violations), enabling speed-based decision workflows
- **Before notification blocks** to alert on speed violations or threshold events (e.g., alert on speeding violations, notify on rapid movement, trigger speed-based alerts), enabling velocity-based notification workflows
- **Before data storage blocks** to record velocity measurements (e.g., log speed data, store velocity statistics, record movement metrics), enabling velocity data logging workflows

## Requirements

This block requires tracked detections with tracker_id information (detections must come from a tracking block like Byte Tracker). The image's video_metadata should include frame rate (fps) for video files or frame timestamps for streamed video to calculate accurate time deltas. The block maintains persistent tracking state across frames for each video using video_identifier, so it should be used in video workflows where frames are processed sequentially. For accurate velocity measurement, detections should be provided consistently across frames with valid tracker IDs. The pixels_per_meter conversion factor should be calibrated based on camera setup and scene geometry for accurate real-world speed measurements. Note that velocity accuracy may vary due to perspective distortion depending on object position in the frame.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/velocity@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `smoothing_alpha` | `float` | Smoothing factor (alpha) for exponential moving average, range 0 < alpha <= 1. Controls how much smoothing is applied to velocity measurements. Lower values (closer to 0) provide more smoothing - slower response to changes, less noise. Higher values (closer to 1) provide less smoothing - faster response to changes, more noise. Default 0.5 balances smoothness and responsiveness.. | ✅ |
| `pixels_per_meter` | `float` | Conversion factor from pixels to meters for real-world speed calculation. Velocity measurements in pixels per second are divided by this value to convert to meters per second. Must be greater than 0. For accurate real-world speeds, calibrate based on camera height, angle, and scene geometry. Example: if 1 pixel = 0.01 meters (1cm), use 0.01. Default 1.0 means no conversion (results in pixels per second).. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Velocity` in version `v1`.

    - inputs: [`Detections Stitch`](detections_stitch.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Seg Preview`](seg_preview.md), [`Detections Filter`](detections_filter.md), [`Object Detection Model`](object_detection_model.md), [`Dynamic Zone`](dynamic_zone.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Gaze Detection`](gaze_detection.md), [`OCR Model`](ocr_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Google Vision OCR`](google_vision_ocr.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`EasyOCR`](easy_ocr.md), [`Detections Combine`](detections_combine.md), [`Object Detection Model`](object_detection_model.md), [`Motion Detection`](motion_detection.md), [`Cosine Similarity`](cosine_similarity.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detection Event Log`](detection_event_log.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Bounding Rectangle`](bounding_rectangle.md), [`VLM As Detector`](vlm_as_detector.md), [`Byte Tracker`](byte_tracker.md), [`Detections Consensus`](detections_consensus.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Time in Zone`](timein_zone.md), [`Detections Merge`](detections_merge.md), [`Perspective Correction`](perspective_correction.md), [`Overlap Filter`](overlap_filter.md), [`Velocity`](velocity.md), [`Object Detection Model`](object_detection_model.md), [`Identify Changes`](identify_changes.md), [`Camera Focus`](camera_focus.md), [`Byte Tracker`](byte_tracker.md), [`YOLO-World Model`](yolo_world_model.md), [`Detection Offset`](detection_offset.md), [`SAM 3`](sam3.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`SAM 3`](sam3.md), [`VLM As Detector`](vlm_as_detector.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Template Matching`](template_matching.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Dynamic Crop`](dynamic_crop.md), [`SORT Tracker`](sort_tracker.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Moondream2`](moondream2.md), [`Camera Focus`](camera_focus.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Detections Transformation`](detections_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Path Deviation`](path_deviation.md), [`Time in Zone`](timein_zone.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Path Deviation`](path_deviation.md), [`SAM 3`](sam3.md)
    - outputs: [`Detections Stabilizer`](detections_stabilizer.md), [`Detections Stitch`](detections_stitch.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Distance Measurement`](distance_measurement.md), [`Color Visualization`](color_visualization.md), [`Detections Combine`](detections_combine.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Ellipse Visualization`](ellipse_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Polygon Visualization`](polygon_visualization.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Byte Tracker`](byte_tracker.md), [`Detections Consensus`](detections_consensus.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Time in Zone`](timein_zone.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Camera Focus`](camera_focus.md), [`Detection Offset`](detection_offset.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Size Measurement`](size_measurement.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Florence-2 Model`](florence2_model.md), [`Halo Visualization`](halo_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Crop Visualization`](crop_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Path Deviation`](path_deviation.md), [`Time in Zone`](timein_zone.md), [`Dot Visualization`](dot_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Path Deviation`](path_deviation.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Detections Filter`](detections_filter.md), [`Icon Visualization`](icon_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Dynamic Zone`](dynamic_zone.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Blur Visualization`](blur_visualization.md), [`Detections Merge`](detections_merge.md), [`Perspective Correction`](perspective_correction.md), [`Overlap Filter`](overlap_filter.md), [`Line Counter`](line_counter.md), [`Velocity`](velocity.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Label Visualization`](label_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Stitch OCR Detections`](stitch_ocr_detections.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Velocity` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Image with embedded video metadata. The video_metadata contains fps, frame_number, frame_timestamp, and video_identifier. Required for calculating time deltas and maintaining separate velocity tracking state for different videos..
        - `detections` (*Union[[`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Tracked object detection or instance segmentation predictions. Must include tracker_id information from a tracking block. Velocity is calculated based on displacement of bounding box centers over time. Output detections include velocity (m/s vector), speed (m/s scalar), smoothed_velocity (m/s vector), and smoothed_speed (m/s scalar) in detection metadata..
        - `smoothing_alpha` (*[`float`](../kinds/float.md)*): Smoothing factor (alpha) for exponential moving average, range 0 < alpha <= 1. Controls how much smoothing is applied to velocity measurements. Lower values (closer to 0) provide more smoothing - slower response to changes, less noise. Higher values (closer to 1) provide less smoothing - faster response to changes, more noise. Default 0.5 balances smoothness and responsiveness..
        - `pixels_per_meter` (*[`float`](../kinds/float.md)*): Conversion factor from pixels to meters for real-world speed calculation. Velocity measurements in pixels per second are divided by this value to convert to meters per second. Must be greater than 0. For accurate real-world speeds, calibrate based on camera height, angle, and scene geometry. Example: if 1 pixel = 0.01 meters (1cm), use 0.01. Default 1.0 means no conversion (results in pixels per second)..

    - output
    
        - `velocity_detections` (*Union[[`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Prediction with detected bounding boxes and detected keypoints in form of sv.Detections(...) object if `keypoint_detection_prediction` or Prediction with detected bounding boxes in form of sv.Detections(...) object if `object_detection_prediction` or Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object if `instance_segmentation_prediction`.



??? tip "Example JSON definition of step `Velocity` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/velocity@v1",
	    "image": "<block_does_not_provide_example>",
	    "detections": "$steps.object_detection_model.predictions",
	    "smoothing_alpha": 0.5,
	    "pixels_per_meter": 0.01
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    


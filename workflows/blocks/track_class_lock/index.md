
# Track Class Lock



??? "Class: `TrackClassLockBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/transformations/track_class_lock/v1.py">inference.core.workflows.core_steps.transformations.track_class_lock.v1.TrackClassLockBlockV1</a>
    



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


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/track_class_lock@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `min_votes` | `int` | Cumulative qualifying votes a class needs before the initial lock is acquired. Higher values delay locking but make the initial decision more reliable.. | ✅ |
| `vote_confidence` | `float` | Minimum prediction confidence for a frame to count, both for pre-lock votes and post-lock challenger streaks. Frames below this threshold are ignored.. | ✅ |
| `lead_margin` | `int` | Number of votes by which the top class must lead the runner-up before locking. Prevents premature locks when two classes are contested.. | ✅ |
| `switch_after` | `int` | Number of CONSECUTIVE qualifying frames of the same challenger class required to change an existing lock. Any interruption resets the streak. Minimum 1 (a value of 1 switches on a single contrary frame; use >= 2 to enforce a multi-frame streak).. | ✅ |
| `state_ttl` | `int` | Number of frames after which state of unseen tracks is purged.. | ✅ |
| `reattach_window` | `int` | When a NEW tracker id appears where a locked track disappeared within this many frames, the new track inherits the lost track's lock and votes. Bridges tracker id switches caused by short detection gaps. Set to 0 to disable re-attachment.. | ✅ |
| `reattach_iou` | `float` | Minimum IoU between a new detection's bounding box and a recently lost locked track's last known bounding box for the lock to be inherited. Higher values require the object to reappear closer to where it vanished.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-alert-circle-outline:{ style="color: #f57c00" } `soft` — runtime `hosted_serverless`, `dedicated_deployment`; execution `remote`; input `video`
:   Block keeps per-video state in process memory (keyed by video_metadata.video_identifier). With remote step execution on stateless or multi-replica HTTP runtimes, successive requests may be served by different worker processes, so the state resets between calls and the output is meaningless for tracking / counting / aggregation. Use local step execution in an InferencePipeline for stable cross-frame results.

:material-alert-circle-outline:{ style="color: #f57c00" } `soft` — input `image`
:   Block depends on temporal context from video or repeated-frame workflows. With a still image/photo, there is no meaningful history to track, compare, aggregate, or visualize, so the block provides little or no benefit.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Track Class Lock` in version `v1`.

    - inputs: [`Halo Visualization`](halo_visualization.md), [`Image Threshold`](image_threshold.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Template Matching`](template_matching.md), [`Stitch Images`](stitch_images.md), [`Morphological Transformation`](morphological_transformation.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Blur Visualization`](blur_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Reference Path Visualization`](reference_path_visualization.md), [`YOLO-World Model`](yolo_world_model.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Byte Tracker`](byte_tracker.md), [`Camera Focus`](camera_focus.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`QR Code Generator`](qr_code_generator.md), [`Track Class Lock`](track_class_lock.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Path Deviation`](path_deviation.md), [`Trace Visualization`](trace_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Dot Visualization`](dot_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Label Visualization`](label_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Seg Preview`](seg_preview.md), [`Text Display`](text_display.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Image Blur`](image_blur.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Detections Merge`](detections_merge.md), [`Velocity`](velocity.md), [`SIFT`](sift.md), [`Gaze Detection`](gaze_detection.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`EasyOCR`](easy_ocr.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`SAM 3`](sam3.md), [`Triangle Visualization`](triangle_visualization.md), [`Camera Focus`](camera_focus.md), [`Contrast Equalization`](contrast_equalization.md), [`Time in Zone`](timein_zone.md), [`Polygon Visualization`](polygon_visualization.md), [`SORT Tracker`](sort_tracker.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`VLM As Detector`](vlm_as_detector.md), [`Line Counter`](line_counter.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Clip Comparison`](clip_comparison.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Halo Visualization`](halo_visualization.md), [`Color Visualization`](color_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`VLM As Detector`](vlm_as_detector.md), [`Identify Changes`](identify_changes.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Stack`](image_stack.md), [`Mask Visualization`](mask_visualization.md), [`Detections Filter`](detections_filter.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Distance Measurement`](distance_measurement.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Time in Zone`](timein_zone.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Keypoint Visualization`](keypoint_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Overlap Filter`](overlap_filter.md), [`Identify Outliers`](identify_outliers.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Image Slicer`](image_slicer.md), [`Detection Offset`](detection_offset.md), [`Image Contours`](image_contours.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Detections Consensus`](detections_consensus.md), [`Object Detection Model`](object_detection_model.md), [`Image Preprocessing`](image_preprocessing.md), [`Byte Tracker`](byte_tracker.md), [`SAM 3`](sam3.md), [`Dynamic Crop`](dynamic_crop.md), [`Path Deviation`](path_deviation.md), [`Byte Tracker`](byte_tracker.md), [`Depth Estimation`](depth_estimation.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Detections Combine`](detections_combine.md), [`Pixel Color Count`](pixel_color_count.md), [`Motion Detection`](motion_detection.md), [`Corner Visualization`](corner_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Moondream2`](moondream2.md), [`Grid Visualization`](grid_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`SAM 3`](sam3.md), [`Circle Visualization`](circle_visualization.md), [`Image Slicer`](image_slicer.md), [`SIFT Comparison`](sift_comparison.md), [`Time in Zone`](timein_zone.md), [`OCR Model`](ocr_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Detection Event Log`](detection_event_log.md), [`SIFT Comparison`](sift_comparison.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`Background Color Visualization`](background_color_visualization.md), [`Line Counter`](line_counter.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Dynamic Zone`](dynamic_zone.md)
    - outputs: [`Overlap Analysis`](overlap_analysis.md), [`Halo Visualization`](halo_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Crop Visualization`](crop_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Blur Visualization`](blur_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Byte Tracker`](byte_tracker.md), [`Track Class Lock`](track_class_lock.md), [`Size Measurement`](size_measurement.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Path Deviation`](path_deviation.md), [`Trace Visualization`](trace_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Dot Visualization`](dot_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Label Visualization`](label_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Detections Merge`](detections_merge.md), [`Velocity`](velocity.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Triangle Visualization`](triangle_visualization.md), [`Camera Focus`](camera_focus.md), [`Time in Zone`](timein_zone.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`SORT Tracker`](sort_tracker.md), [`Line Counter`](line_counter.md), [`Polygon Visualization`](polygon_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Halo Visualization`](halo_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Color Visualization`](color_visualization.md), [`Event Writer`](event_writer.md), [`Polygon Visualization`](polygon_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Detections Filter`](detections_filter.md), [`Distance Measurement`](distance_measurement.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Bounding Rectangle`](bounding_rectangle.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Time in Zone`](timein_zone.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Overlap Filter`](overlap_filter.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Detection Offset`](detection_offset.md), [`Detections Consensus`](detections_consensus.md), [`Byte Tracker`](byte_tracker.md), [`Path Deviation`](path_deviation.md), [`Dynamic Crop`](dynamic_crop.md), [`Byte Tracker`](byte_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Detections Combine`](detections_combine.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Corner Visualization`](corner_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Circle Visualization`](circle_visualization.md), [`Time in Zone`](timein_zone.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Detection Event Log`](detection_event_log.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Line Counter`](line_counter.md), [`Dynamic Zone`](dynamic_zone.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Track Class Lock` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Image with embedded video metadata. The video_metadata contains video_identifier used to maintain separate voting state for different videos..
        - `detections` (*Union[[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`rle_instance_segmentation_prediction`](../kinds/rle_instance_segmentation_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md)]*): Tracked predictions (object detection, instance segmentation, keypoint detection or RLE instance segmentation). Must include tracker_id information from a tracking block..
        - `min_votes` (*[`integer`](../kinds/integer.md)*): Cumulative qualifying votes a class needs before the initial lock is acquired. Higher values delay locking but make the initial decision more reliable..
        - `vote_confidence` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Minimum prediction confidence for a frame to count, both for pre-lock votes and post-lock challenger streaks. Frames below this threshold are ignored..
        - `lead_margin` (*[`integer`](../kinds/integer.md)*): Number of votes by which the top class must lead the runner-up before locking. Prevents premature locks when two classes are contested..
        - `switch_after` (*[`integer`](../kinds/integer.md)*): Number of CONSECUTIVE qualifying frames of the same challenger class required to change an existing lock. Any interruption resets the streak. Minimum 1 (a value of 1 switches on a single contrary frame; use >= 2 to enforce a multi-frame streak)..
        - `state_ttl` (*[`integer`](../kinds/integer.md)*): Number of frames after which state of unseen tracks is purged..
        - `reattach_window` (*[`integer`](../kinds/integer.md)*): When a NEW tracker id appears where a locked track disappeared within this many frames, the new track inherits the lost track's lock and votes. Bridges tracker id switches caused by short detection gaps. Set to 0 to disable re-attachment..
        - `reattach_iou` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Minimum IoU between a new detection's bounding box and a recently lost locked track's last known bounding box for the lock to be inherited. Higher values require the object to reappear closer to where it vanished..

    - output
    
        - `tracked_detections` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`rle_instance_segmentation_prediction`](../kinds/rle_instance_segmentation_prediction.md)]*): Prediction with detected bounding boxes in form of sv.Detections(...) object if `object_detection_prediction` or Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object if `instance_segmentation_prediction` or Prediction with detected bounding boxes and detected keypoints in form of sv.Detections(...) object if `keypoint_detection_prediction` or Prediction with detected bounding boxes and RLE-encoded segmentation masks in form of sv.Detections(...) object if `rle_instance_segmentation_prediction`.



??? tip "Example JSON definition of step `Track Class Lock` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/track_class_lock@v1",
	    "image": "<block_does_not_provide_example>",
	    "detections": "$steps.byte_tracker.tracked_detections",
	    "min_votes": 10,
	    "vote_confidence": 0.8,
	    "lead_margin": 3,
	    "switch_after": 15,
	    "state_ttl": 300,
	    "reattach_window": 30,
	    "reattach_iou": 0.3
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    


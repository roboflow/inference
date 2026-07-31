
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

    - inputs: [`Image Blur`](image_blur.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Blur Visualization`](blur_visualization.md), [`Track Class Lock`](track_class_lock.md), [`Byte Tracker`](byte_tracker.md), [`Frame Delay`](frame_delay.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Crop Visualization`](crop_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Path Deviation`](path_deviation.md), [`Polygon Visualization`](polygon_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Detections Stitch`](detections_stitch.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Image Slicer`](image_slicer.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Time in Zone`](timein_zone.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Template Matching`](template_matching.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Perspective Correction`](perspective_correction.md), [`Motion Detection`](motion_detection.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`YOLO-World Model`](yolo_world_model.md), [`Halo Visualization`](halo_visualization.md), [`SIFT`](sift.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Label Visualization`](label_visualization.md), [`Overlap Filter`](overlap_filter.md), [`Dynamic Crop`](dynamic_crop.md), [`Detections Consensus`](detections_consensus.md), [`Label Visualization`](label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OCR Model`](ocr_model.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Halo Visualization`](halo_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Time in Zone`](timein_zone.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Velocity`](velocity.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Time in Zone`](timein_zone.md), [`Detection Event Log`](detection_event_log.md), [`Pixel Color Count`](pixel_color_count.md), [`Background Subtraction`](background_subtraction.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Stitch Images`](stitch_images.md), [`Path Deviation`](path_deviation.md), [`Relative Static Crop`](relative_static_crop.md), [`Corner Visualization`](corner_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Depth Estimation`](depth_estimation.md), [`Detections Combine`](detections_combine.md), [`Detection Offset`](detection_offset.md), [`SAM 3`](sam3.md), [`Dot Visualization`](dot_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Bounding Rectangle`](bounding_rectangle.md), [`SIFT Comparison`](sift_comparison.md), [`Distance Measurement`](distance_measurement.md), [`Camera Calibration`](camera_calibration.md), [`Seg Preview`](seg_preview.md), [`EasyOCR`](easy_ocr.md), [`Background Color Visualization`](background_color_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Image Contours`](image_contours.md), [`Camera Focus`](camera_focus.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`PP-OCR`](ppocr.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Contrast Equalization`](contrast_equalization.md), [`Morphological Transformation`](morphological_transformation.md), [`SIFT Comparison`](sift_comparison.md), [`Image Threshold`](image_threshold.md), [`Trace Visualization`](trace_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Color Visualization`](color_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Camera Focus`](camera_focus.md), [`SAM 3`](sam3.md), [`Line Counter`](line_counter.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Circle Visualization`](circle_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Morphological Transformation`](morphological_transformation.md), [`Dynamic Zone`](dynamic_zone.md), [`Detections Merge`](detections_merge.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Google Vision OCR`](google_vision_ocr.md), [`Detections Filter`](detections_filter.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`Polygon Visualization`](polygon_visualization.md), [`Icon Visualization`](icon_visualization.md), [`SAM 3`](sam3.md), [`Detections Transformation`](detections_transformation.md), [`Rich Label Visualization`](rich_label_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Identify Changes`](identify_changes.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Image Slicer`](image_slicer.md), [`Gaze Detection`](gaze_detection.md), [`Object Detection Model`](object_detection_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Line Counter`](line_counter.md), [`Clip Comparison`](clip_comparison.md), [`Text Display`](text_display.md), [`Image Stack`](image_stack.md), [`Grid Visualization`](grid_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Identify Outliers`](identify_outliers.md), [`Mask Visualization`](mask_visualization.md), [`Moondream2`](moondream2.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Nearest Neighbor Detection Match`](nearest_neighbor_detection_match.md), [`Byte Tracker`](byte_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`VLM As Detector`](vlm_as_detector.md)
    - outputs: [`Track Class Lock`](track_class_lock.md), [`Byte Tracker`](byte_tracker.md), [`Blur Visualization`](blur_visualization.md), [`Frame Delay`](frame_delay.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Path Deviation`](path_deviation.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Crop Visualization`](crop_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Event Writer`](event_writer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Time in Zone`](timein_zone.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Perspective Correction`](perspective_correction.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Halo Visualization`](halo_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Label Visualization`](label_visualization.md), [`Overlap Filter`](overlap_filter.md), [`Dynamic Crop`](dynamic_crop.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Detections Consensus`](detections_consensus.md), [`Label Visualization`](label_visualization.md), [`GeoTag Detection`](geo_tag_detection.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Halo Visualization`](halo_visualization.md), [`Time in Zone`](timein_zone.md), [`Velocity`](velocity.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Time in Zone`](timein_zone.md), [`Detection Event Log`](detection_event_log.md), [`Overlap Analysis`](overlap_analysis.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Size Measurement`](size_measurement.md), [`Path Deviation`](path_deviation.md), [`Corner Visualization`](corner_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Detections Combine`](detections_combine.md), [`Detection Offset`](detection_offset.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Distance Measurement`](distance_measurement.md), [`Dot Visualization`](dot_visualization.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Florence-2 Model`](florence2_model.md), [`Background Color Visualization`](background_color_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Trace Visualization`](trace_visualization.md), [`Color Visualization`](color_visualization.md), [`Camera Focus`](camera_focus.md), [`Line Counter`](line_counter.md), [`Circle Visualization`](circle_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Detections Merge`](detections_merge.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Detections Filter`](detections_filter.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Rich Label Visualization`](rich_label_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Line Counter`](line_counter.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Mask Visualization`](mask_visualization.md), [`Nearest Neighbor Detection Match`](nearest_neighbor_detection_match.md), [`Byte Tracker`](byte_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Pixelate Visualization`](pixelate_visualization.md)

    
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


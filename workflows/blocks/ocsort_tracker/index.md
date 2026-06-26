
# OC-SORT Tracker



??? "Class: `OCSORTBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/trackers/ocsort/v1.py">inference.core.workflows.core_steps.trackers.ocsort.v1.OCSORTBlockV1</a>
    



Track objects across video frames using the **OC-SORT** algorithm from the
roboflow/trackers package.

OC-SORT extends SORT with two key mechanisms:

1. **Observation-Centric Re-Update (OCR):** When a track reappears after
   occlusion, OC-SORT retroactively corrects the Kalman filter using the real
   observations before and after the gap, reducing accumulated drift.
2. **Observation-Centric Momentum (OCM):** A direction-consistency cost is
   blended with IoU during association, penalising matches where the candidate
   detection lies in a direction inconsistent with the track's recent motion.

This makes OC-SORT significantly more robust than SORT in scenes with heavy
occlusion, erratic motion, and uniform appearance.

**When to use OC-SORT:**
- Crowded scenes with frequent and prolonged occlusions (e.g. pedestrians,
  warehouse workers).
- Non-linear or erratic motion patterns (e.g. dancing, sports with abrupt
  direction changes).
- When identity consistency over long sequences is more important than raw speed.

**When to consider alternatives:**
- For general-purpose tracking with mixed-confidence detections, try **ByteTrack**.
- For maximum simplicity and speed with a strong detector, try **SORT**.

Outputs three detection sets:
- **tracked_detections**: All confirmed tracked detections with assigned track IDs.
- **new_instances**: Detections whose track ID appears for the first time.
- **already_seen_instances**: Detections whose track ID has been seen in a prior frame.

The block maintains separate tracker state and instance cache per `video_identifier`,
enabling multi-stream tracking within a single workflow.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/trackers_ocsort@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `minimum_iou_threshold` | `float` | Minimum IoU required to associate a detection with an existing track. Default: 0.3.. | ✅ |
| `minimum_consecutive_frames` | `int` | Number of consecutive frames a track must be matched before it is emitted as a confirmed track (tracker_id != -1). Default: 3.. | ✅ |
| `lost_track_buffer` | `int` | Number of frames to keep a track alive after it loses its matched detection. Higher values improve occlusion recovery. Default: 30.. | ✅ |
| `high_conf_det_threshold` | `float` | Confidence threshold for high-confidence detections used in association. Default: 0.6.. | ✅ |
| `direction_consistency_weight` | `float` | Weight for the direction consistency term in the OC-SORT association cost. Higher values prioritise alignment between historical motion direction and the direction to the candidate detection. Default: 0.2.. | ✅ |
| `delta_t` | `int` | Number of past frames used by OC-SORT to estimate per-track velocity for direction consistency momentum. Default: 3.. | ✅ |
| `instances_cache_size` | `int` | Maximum number of track IDs retained in the instance cache for new/already-seen categorisation. Uses FIFO eviction. Default: 16384.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-alert-circle-outline:{ style="color: #f57c00" } `soft` — runtime `hosted_serverless`, `dedicated_deployment`; execution `remote`; input `video`
:   Block keeps per-video state in process memory (keyed by video_metadata.video_identifier). With remote step execution on stateless or multi-replica HTTP runtimes, successive requests may be served by different worker processes, so the state resets between calls and the output is meaningless for tracking / counting / aggregation. Use local step execution in an InferencePipeline for stable cross-frame results.

:material-alert-circle-outline:{ style="color: #f57c00" } `soft` — input `image`
:   Block depends on temporal context from video or repeated-frame workflows. With a still image/photo, there is no meaningful history to track, compare, aggregate, or visualize, so the block provides little or no benefit.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `OC-SORT Tracker` in version `v1`.

    - inputs: [`YOLO-World Model`](yolo_world_model.md), [`SAM 3`](sam3.md), [`Dynamic Zone`](dynamic_zone.md), [`Byte Tracker`](byte_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`SORT Tracker`](sort_tracker.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Clip Comparison`](clip_comparison.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Detections Consensus`](detections_consensus.md), [`Time in Zone`](timein_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`OCR Model`](ocr_model.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Image Threshold`](image_threshold.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Slicer`](image_slicer.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Image Blur`](image_blur.md), [`Depth Estimation`](depth_estimation.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Camera Focus`](camera_focus.md), [`Crop Visualization`](crop_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Time in Zone`](timein_zone.md), [`Label Visualization`](label_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Track Class Lock`](track_class_lock.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Text Display`](text_display.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Velocity`](velocity.md), [`Identify Outliers`](identify_outliers.md), [`Corner Visualization`](corner_visualization.md), [`Image Slicer`](image_slicer.md), [`Distance Measurement`](distance_measurement.md), [`Image Preprocessing`](image_preprocessing.md), [`Morphological Transformation`](morphological_transformation.md), [`Triangle Visualization`](triangle_visualization.md), [`SAM 3`](sam3.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Identify Changes`](identify_changes.md), [`Halo Visualization`](halo_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Background Subtraction`](background_subtraction.md), [`Path Deviation`](path_deviation.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Moondream2`](moondream2.md), [`Motion Detection`](motion_detection.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Detection Event Log`](detection_event_log.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Detections Combine`](detections_combine.md), [`SIFT`](sift.md), [`Byte Tracker`](byte_tracker.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Vision OCR`](google_vision_ocr.md), [`SIFT Comparison`](sift_comparison.md), [`Line Counter`](line_counter.md), [`Overlap Filter`](overlap_filter.md), [`Grid Visualization`](grid_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Reference Path Visualization`](reference_path_visualization.md), [`EasyOCR`](easy_ocr.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Template Matching`](template_matching.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Seg Preview`](seg_preview.md), [`Stitch Images`](stitch_images.md), [`Line Counter`](line_counter.md), [`Circle Visualization`](circle_visualization.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Object Detection Model`](object_detection_model.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Image Stack`](image_stack.md), [`Relative Static Crop`](relative_static_crop.md), [`Gaze Detection`](gaze_detection.md), [`Detections Merge`](detections_merge.md), [`SIFT Comparison`](sift_comparison.md), [`Path Deviation`](path_deviation.md), [`Detection Offset`](detection_offset.md), [`VLM As Detector`](vlm_as_detector.md), [`QR Code Generator`](qr_code_generator.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Dot Visualization`](dot_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Camera Focus`](camera_focus.md), [`Pixel Color Count`](pixel_color_count.md), [`Color Visualization`](color_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Dynamic Crop`](dynamic_crop.md), [`Halo Visualization`](halo_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Morphological Transformation`](morphological_transformation.md), [`Polygon Visualization`](polygon_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Icon Visualization`](icon_visualization.md), [`Detections Filter`](detections_filter.md), [`Detections Transformation`](detections_transformation.md), [`Background Color Visualization`](background_color_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Image Contours`](image_contours.md), [`SAM2 Video Tracker`](sam2_video_tracker.md)
    - outputs: [`Dynamic Zone`](dynamic_zone.md), [`Florence-2 Model`](florence2_model.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`SORT Tracker`](sort_tracker.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Byte Tracker`](byte_tracker.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Detections Consensus`](detections_consensus.md), [`Time in Zone`](timein_zone.md), [`Time in Zone`](timein_zone.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Byte Tracker`](byte_tracker.md), [`Polygon Visualization`](polygon_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Time in Zone`](timein_zone.md), [`Label Visualization`](label_visualization.md), [`Track Class Lock`](track_class_lock.md), [`Velocity`](velocity.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Corner Visualization`](corner_visualization.md), [`Distance Measurement`](distance_measurement.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Triangle Visualization`](triangle_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Halo Visualization`](halo_visualization.md), [`Path Deviation`](path_deviation.md), [`Detection Event Log`](detection_event_log.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Detections Combine`](detections_combine.md), [`Byte Tracker`](byte_tracker.md), [`Line Counter`](line_counter.md), [`Size Measurement`](size_measurement.md), [`Overlap Filter`](overlap_filter.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Line Counter`](line_counter.md), [`Circle Visualization`](circle_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Event Writer`](event_writer.md), [`Detections Merge`](detections_merge.md), [`Path Deviation`](path_deviation.md), [`Detection Offset`](detection_offset.md), [`Dot Visualization`](dot_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Camera Focus`](camera_focus.md), [`Perspective Correction`](perspective_correction.md), [`Blur Visualization`](blur_visualization.md), [`Overlap Analysis`](overlap_analysis.md), [`Dynamic Crop`](dynamic_crop.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Color Visualization`](color_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Florence-2 Model`](florence2_model.md), [`Halo Visualization`](halo_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Detections Filter`](detections_filter.md), [`Icon Visualization`](icon_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Background Color Visualization`](background_color_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Ellipse Visualization`](ellipse_visualization.md), [`SAM2 Video Tracker`](sam2_video_tracker.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`OC-SORT Tracker` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Input image with embedded video metadata (fps and video_identifier). Used to initialise and retrieve per-video tracker state..
        - `detections` (*Union[[`rle_instance_segmentation_prediction`](../kinds/rle_instance_segmentation_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Detection predictions for the current frame to track..
        - `minimum_iou_threshold` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Minimum IoU required to associate a detection with an existing track. Default: 0.3..
        - `minimum_consecutive_frames` (*[`integer`](../kinds/integer.md)*): Number of consecutive frames a track must be matched before it is emitted as a confirmed track (tracker_id != -1). Default: 3..
        - `lost_track_buffer` (*[`integer`](../kinds/integer.md)*): Number of frames to keep a track alive after it loses its matched detection. Higher values improve occlusion recovery. Default: 30..
        - `high_conf_det_threshold` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Confidence threshold for high-confidence detections used in association. Default: 0.6..
        - `direction_consistency_weight` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Weight for the direction consistency term in the OC-SORT association cost. Higher values prioritise alignment between historical motion direction and the direction to the candidate detection. Default: 0.2..
        - `delta_t` (*[`integer`](../kinds/integer.md)*): Number of past frames used by OC-SORT to estimate per-track velocity for direction consistency momentum. Default: 3..

    - output
    
        - `tracked_detections` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`rle_instance_segmentation_prediction`](../kinds/rle_instance_segmentation_prediction.md)]*): Prediction with detected bounding boxes in form of sv.Detections(...) object if `object_detection_prediction` or Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object if `instance_segmentation_prediction` or Prediction with detected bounding boxes and detected keypoints in form of sv.Detections(...) object if `keypoint_detection_prediction` or Prediction with detected bounding boxes and RLE-encoded segmentation masks in form of sv.Detections(...) object if `rle_instance_segmentation_prediction`.
        - `new_instances` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`rle_instance_segmentation_prediction`](../kinds/rle_instance_segmentation_prediction.md)]*): Prediction with detected bounding boxes in form of sv.Detections(...) object if `object_detection_prediction` or Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object if `instance_segmentation_prediction` or Prediction with detected bounding boxes and detected keypoints in form of sv.Detections(...) object if `keypoint_detection_prediction` or Prediction with detected bounding boxes and RLE-encoded segmentation masks in form of sv.Detections(...) object if `rle_instance_segmentation_prediction`.
        - `already_seen_instances` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`rle_instance_segmentation_prediction`](../kinds/rle_instance_segmentation_prediction.md)]*): Prediction with detected bounding boxes in form of sv.Detections(...) object if `object_detection_prediction` or Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object if `instance_segmentation_prediction` or Prediction with detected bounding boxes and detected keypoints in form of sv.Detections(...) object if `keypoint_detection_prediction` or Prediction with detected bounding boxes and RLE-encoded segmentation masks in form of sv.Detections(...) object if `rle_instance_segmentation_prediction`.



??? tip "Example JSON definition of step `OC-SORT Tracker` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/trackers_ocsort@v1",
	    "image": "<block_does_not_provide_example>",
	    "detections": "$steps.object_detection_model.predictions",
	    "minimum_iou_threshold": 0.3,
	    "minimum_consecutive_frames": 3,
	    "lost_track_buffer": 30,
	    "high_conf_det_threshold": 0.6,
	    "direction_consistency_weight": 0.2,
	    "delta_t": 3,
	    "instances_cache_size": "<block_does_not_provide_example>"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    


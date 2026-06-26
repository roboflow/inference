
# ByteTrack Tracker



??? "Class: `ByteTrackBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/trackers/bytetrack/v1.py">inference.core.workflows.core_steps.trackers.bytetrack.v1.ByteTrackBlockV1</a>
    



Track objects across video frames using the **ByteTrack** algorithm from the
roboflow/trackers package.

ByteTrack splits detections into high- and low-confidence pools and runs two
rounds of IoU-based association.  The first round matches high-confidence
detections to existing tracks; the second recovers weak detections that overlap
unmatched tracks.  This makes ByteTrack particularly effective in **dense
environments** where objects are frequently partially occluded and detector
confidence fluctuates.

**When to use ByteTrack:**
- General-purpose tracking across diverse scenes.
- Dense or crowded environments with partial occlusions.
- Sports tracking and fast-moving objects (highest benchmark scores on SportsMOT).
- When your detector produces a mix of high- and low-confidence detections that
  you want to retain.

**When to consider alternatives:**
- For maximum simplicity and speed with a strong detector, use **SORT**.
- For scenes with heavy occlusion and non-linear motion, use **OC-SORT**.

Outputs three detection sets:
- **tracked_detections**: All confirmed tracked detections with assigned track IDs.
- **new_instances**: Detections whose track ID appears for the first time.
- **already_seen_instances**: Detections whose track ID has been seen in a prior frame.

The block maintains separate tracker state and instance cache per `video_identifier`,
enabling multi-stream tracking within a single workflow.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/trackers_bytetrack@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `minimum_iou_threshold` | `float` | Minimum IoU required to associate a detection with an existing track. Default: 0.1.. | ✅ |
| `minimum_consecutive_frames` | `int` | Number of consecutive frames a track must be matched before it is emitted as a confirmed track (tracker_id != -1). Default: 2.. | ✅ |
| `lost_track_buffer` | `int` | Number of frames to keep a track alive after it loses its matched detection. Higher values improve occlusion recovery. Default: 30.. | ✅ |
| `track_activation_threshold` | `float` | Minimum detection confidence required to spawn a new track. Detections below this threshold are not used to create new tracks. Default: 0.7.. | ✅ |
| `high_conf_det_threshold` | `float` | Confidence threshold for high-confidence detections used in association. Default: 0.6.. | ✅ |
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
    Check what blocks you can connect to `ByteTrack Tracker` in version `v1`.

    - inputs: [`Image Blur`](image_blur.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Google Vision OCR`](google_vision_ocr.md), [`Byte Tracker`](byte_tracker.md), [`Image Threshold`](image_threshold.md), [`Morphological Transformation`](morphological_transformation.md), [`VLM As Detector`](vlm_as_detector.md), [`Triangle Visualization`](triangle_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Detections Combine`](detections_combine.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Detections Merge`](detections_merge.md), [`Grid Visualization`](grid_visualization.md), [`Mask Visualization`](mask_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Camera Focus`](camera_focus.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections Transformation`](detections_transformation.md), [`Template Matching`](template_matching.md), [`Track Class Lock`](track_class_lock.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Dynamic Zone`](dynamic_zone.md), [`SAM 3`](sam3.md), [`Dynamic Crop`](dynamic_crop.md), [`Velocity`](velocity.md), [`SAM 3`](sam3.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`QR Code Generator`](qr_code_generator.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Color Visualization`](color_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`YOLO-World Model`](yolo_world_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Halo Visualization`](halo_visualization.md), [`Label Visualization`](label_visualization.md), [`SIFT`](sift.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Byte Tracker`](byte_tracker.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Detection Offset`](detection_offset.md), [`Object Detection Model`](object_detection_model.md), [`Path Deviation`](path_deviation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Seg Preview`](seg_preview.md), [`Morphological Transformation`](morphological_transformation.md), [`Clip Comparison`](clip_comparison.md), [`Text Display`](text_display.md), [`Overlap Filter`](overlap_filter.md), [`Gaze Detection`](gaze_detection.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Detections Filter`](detections_filter.md), [`Polygon Visualization`](polygon_visualization.md), [`Moondream2`](moondream2.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Background Color Visualization`](background_color_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`EasyOCR`](easy_ocr.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Corner Visualization`](corner_visualization.md), [`Motion Detection`](motion_detection.md), [`SORT Tracker`](sort_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Identify Outliers`](identify_outliers.md), [`Time in Zone`](timein_zone.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Depth Estimation`](depth_estimation.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Time in Zone`](timein_zone.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Mask Area Measurement`](mask_area_measurement.md), [`SIFT Comparison`](sift_comparison.md), [`Image Slicer`](image_slicer.md), [`Image Stack`](image_stack.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Line Counter`](line_counter.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stitch Images`](stitch_images.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Detections Stitch`](detections_stitch.md), [`Byte Tracker`](byte_tracker.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`OCR Model`](ocr_model.md), [`Perspective Correction`](perspective_correction.md), [`Crop Visualization`](crop_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Image Slicer`](image_slicer.md), [`Camera Focus`](camera_focus.md), [`Identify Changes`](identify_changes.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Line Counter`](line_counter.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Distance Measurement`](distance_measurement.md), [`Detections Consensus`](detections_consensus.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Pixel Color Count`](pixel_color_count.md), [`Trace Visualization`](trace_visualization.md), [`SAM 3`](sam3.md), [`Circle Visualization`](circle_visualization.md), [`Time in Zone`](timein_zone.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Image Contours`](image_contours.md), [`Path Deviation`](path_deviation.md), [`Contrast Equalization`](contrast_equalization.md)
    - outputs: [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Triangle Visualization`](triangle_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Detections Combine`](detections_combine.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Detections Merge`](detections_merge.md), [`Florence-2 Model`](florence2_model.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Detections Transformation`](detections_transformation.md), [`Camera Focus`](camera_focus.md), [`Track Class Lock`](track_class_lock.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Halo Visualization`](halo_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Dynamic Crop`](dynamic_crop.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Velocity`](velocity.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Label Visualization`](label_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Detection Offset`](detection_offset.md), [`Florence-2 Model`](florence2_model.md), [`Path Deviation`](path_deviation.md), [`Overlap Filter`](overlap_filter.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Detections Filter`](detections_filter.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Polygon Visualization`](polygon_visualization.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Background Color Visualization`](background_color_visualization.md), [`Overlap Analysis`](overlap_analysis.md), [`Corner Visualization`](corner_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Time in Zone`](timein_zone.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Time in Zone`](timein_zone.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Polygon Visualization`](polygon_visualization.md), [`Line Counter`](line_counter.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Keypoint Visualization`](keypoint_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Byte Tracker`](byte_tracker.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Perspective Correction`](perspective_correction.md), [`Crop Visualization`](crop_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Line Counter`](line_counter.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Distance Measurement`](distance_measurement.md), [`Detections Consensus`](detections_consensus.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Size Measurement`](size_measurement.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Time in Zone`](timein_zone.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Event Writer`](event_writer.md), [`Path Deviation`](path_deviation.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`ByteTrack Tracker` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Input image with embedded video metadata (fps and video_identifier). Used to initialise and retrieve per-video tracker state..
        - `detections` (*Union[[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md), [`rle_instance_segmentation_prediction`](../kinds/rle_instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md)]*): Detection predictions for the current frame to track..
        - `minimum_iou_threshold` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Minimum IoU required to associate a detection with an existing track. Default: 0.1..
        - `minimum_consecutive_frames` (*[`integer`](../kinds/integer.md)*): Number of consecutive frames a track must be matched before it is emitted as a confirmed track (tracker_id != -1). Default: 2..
        - `lost_track_buffer` (*[`integer`](../kinds/integer.md)*): Number of frames to keep a track alive after it loses its matched detection. Higher values improve occlusion recovery. Default: 30..
        - `track_activation_threshold` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Minimum detection confidence required to spawn a new track. Detections below this threshold are not used to create new tracks. Default: 0.7..
        - `high_conf_det_threshold` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Confidence threshold for high-confidence detections used in association. Default: 0.6..

    - output
    
        - `tracked_detections` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`rle_instance_segmentation_prediction`](../kinds/rle_instance_segmentation_prediction.md)]*): Prediction with detected bounding boxes in form of sv.Detections(...) object if `object_detection_prediction` or Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object if `instance_segmentation_prediction` or Prediction with detected bounding boxes and detected keypoints in form of sv.Detections(...) object if `keypoint_detection_prediction` or Prediction with detected bounding boxes and RLE-encoded segmentation masks in form of sv.Detections(...) object if `rle_instance_segmentation_prediction`.
        - `new_instances` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`rle_instance_segmentation_prediction`](../kinds/rle_instance_segmentation_prediction.md)]*): Prediction with detected bounding boxes in form of sv.Detections(...) object if `object_detection_prediction` or Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object if `instance_segmentation_prediction` or Prediction with detected bounding boxes and detected keypoints in form of sv.Detections(...) object if `keypoint_detection_prediction` or Prediction with detected bounding boxes and RLE-encoded segmentation masks in form of sv.Detections(...) object if `rle_instance_segmentation_prediction`.
        - `already_seen_instances` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`rle_instance_segmentation_prediction`](../kinds/rle_instance_segmentation_prediction.md)]*): Prediction with detected bounding boxes in form of sv.Detections(...) object if `object_detection_prediction` or Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object if `instance_segmentation_prediction` or Prediction with detected bounding boxes and detected keypoints in form of sv.Detections(...) object if `keypoint_detection_prediction` or Prediction with detected bounding boxes and RLE-encoded segmentation masks in form of sv.Detections(...) object if `rle_instance_segmentation_prediction`.



??? tip "Example JSON definition of step `ByteTrack Tracker` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/trackers_bytetrack@v1",
	    "image": "<block_does_not_provide_example>",
	    "detections": "$steps.object_detection_model.predictions",
	    "minimum_iou_threshold": 0.1,
	    "minimum_consecutive_frames": 2,
	    "lost_track_buffer": 30,
	    "track_activation_threshold": 0.7,
	    "high_conf_det_threshold": 0.6,
	    "instances_cache_size": "<block_does_not_provide_example>"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    


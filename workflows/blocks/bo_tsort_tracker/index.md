
# BoT-SORT Tracker



??? "Class: `BoTSORTBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/trackers/botsort/v1.py">inference.core.workflows.core_steps.trackers.botsort.v1.BoTSORTBlockV1</a>
    



Track objects across video frames using the **BoT-SORT** algorithm from the
roboflow/trackers package.

BoT-SORT follows a ByteTrack-style association pipeline (high- and low-confidence
detections, Kalman track states) and can apply **camera motion compensation (CMC)**
before association when enabled. CMC estimates a global affine motion between
frames so predicted boxes align better when the camera moves.

**When to use BoT-SORT:**
- Scenes with **moving or shaking cameras** (enable **Camera motion compensation**).
- Dense detection noise where ByteTrack-style two-stage matching helps.
- When you want ByteTrack-like behaviour with an optional motion-compensation stage.

**When to consider alternatives:**
- Fixed camera and you only need speed: **ByteTrack** or **SORT** may be simpler.
- Heavy occlusion and erratic object motion without camera motion: **OC-SORT**.
- Low-texture backgrounds where sparse-feature CMC is unreliable.

**Camera motion compensation:** When enabled, the block passes the workflow image
pixels to the tracker each frame. If the image cannot be decoded to a numpy array,
the tracker runs without CMC for that frame (a warning is logged).

**Instant first-frame activation** defaults to off so behaviour aligns with other
core tracker blocks for ``new_instances`` / ``already_seen_instances``. Enable it
if you want tracks on frame 1 to receive stable IDs immediately (original BoT-SORT
paper-style).

Outputs three detection sets:
- **tracked_detections**: All confirmed tracked detections with assigned track IDs.
- **new_instances**: Detections whose track ID appears for the first time.
- **already_seen_instances**: Detections whose track ID has been seen in a prior frame.

The block maintains separate tracker state and instance cache per `video_identifier`,
enabling multi-stream tracking within a single workflow.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/trackers_botsort@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `minimum_iou_threshold_first_assoc` | `float` | Minimum fused similarity (IoU × confidence) for the first (high-confidence) association step. Default: 0.2.. | ✅ |
| `minimum_iou_threshold_second_assoc` | `float` | Minimum IoU for the second (low-confidence) association step. Default: 0.5.. | ✅ |
| `minimum_iou_threshold_unconfirmed_assoc` | `float` | Minimum fused similarity for matching unconfirmed tracks to remaining high-confidence detections. Default: 0.3.. | ✅ |
| `minimum_consecutive_frames` | `int` | Number of consecutive frames a track must be matched before it is emitted as a confirmed track (tracker_id != -1). Default: 2.. | ✅ |
| `lost_track_buffer` | `int` | Number of frames to keep a track alive after it loses its matched detection. Higher values improve occlusion recovery. Default: 30.. | ✅ |
| `track_activation_threshold` | `float` | Minimum detection confidence required to spawn a new track. Detections below this threshold are not used to create new tracks. Default: 0.7.. | ✅ |
| `high_conf_det_threshold` | `float` | Confidence threshold for high-confidence detections used in association. Default: 0.6.. | ✅ |
| `enable_cmc` | `bool` | Enable camera motion compensation (uses per-frame image pixels). Recommended for moving cameras.. | ✅ |
| `cmc_method` | `str` | Camera motion estimator. One of: orb, sift, sparseOptFlow, ecc. Default: {DEFAULT_CMC_METHOD!r}.. | ❌ |
| `cmc_downscale` | `int` | Downscale factor applied inside CMC for speed and robustness. Default: 2.. | ✅ |
| `instant_first_frame_activation` | `bool` | If true, tracks on the first frame receive IDs immediately (paper-style). Default false so new/already-seen outputs match other core trackers.. | ✅ |
| `instances_cache_size` | `int` | Maximum number of track IDs retained in the instance cache for new/already-seen categorisation. Uses FIFO eviction. Default: 16384.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `BoT-SORT Tracker` in version `v1`.

    - inputs: [`Image Slicer`](image_slicer.md), [`Halo Visualization`](halo_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Perspective Correction`](perspective_correction.md), [`Image Threshold`](image_threshold.md), [`Image Preprocessing`](image_preprocessing.md), [`Mask Edge Snap`](mask_edge_snap.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Circle Visualization`](circle_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Time in Zone`](timein_zone.md), [`Byte Tracker`](byte_tracker.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Detections Stitch`](detections_stitch.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Triangle Visualization`](triangle_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Image Contours`](image_contours.md), [`Object Detection Model`](object_detection_model.md), [`Text Display`](text_display.md), [`SORT Tracker`](sort_tracker.md), [`Local File Sink`](local_file_sink.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Google Vision OCR`](google_vision_ocr.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Camera Focus`](camera_focus.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`EasyOCR`](easy_ocr.md), [`Camera Focus`](camera_focus.md), [`Halo Visualization`](halo_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Line Counter`](line_counter.md), [`Mask Visualization`](mask_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Color Visualization`](color_visualization.md), [`SIFT`](sift.md), [`Detections Combine`](detections_combine.md), [`Line Counter Visualization`](line_counter_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Image Stack`](image_stack.md), [`Path Deviation`](path_deviation.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Time in Zone`](timein_zone.md), [`Webhook Sink`](webhook_sink.md), [`Contrast Equalization`](contrast_equalization.md), [`VLM As Detector`](vlm_as_detector.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Byte Tracker`](byte_tracker.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Email Notification`](email_notification.md), [`Seg Preview`](seg_preview.md), [`Crop Visualization`](crop_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Clip Comparison`](clip_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Detections Filter`](detections_filter.md), [`Moondream2`](moondream2.md), [`Icon Visualization`](icon_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Pixel Color Count`](pixel_color_count.md), [`Morphological Transformation`](morphological_transformation.md), [`JSON Parser`](json_parser.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Background Subtraction`](background_subtraction.md), [`Detections Merge`](detections_merge.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Bounding Rectangle`](bounding_rectangle.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Path Deviation`](path_deviation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Template Matching`](template_matching.md), [`Stitch Images`](stitch_images.md), [`SAM 3`](sam3.md), [`S3 Sink`](s3_sink.md), [`SAM 3`](sam3.md), [`Detections Transformation`](detections_transformation.md), [`Line Counter`](line_counter.md), [`Gaze Detection`](gaze_detection.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Time in Zone`](timein_zone.md), [`Distance Measurement`](distance_measurement.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Image Slicer`](image_slicer.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`VLM As Classifier`](vlm_as_classifier.md), [`YOLO-World Model`](yolo_world_model.md), [`Label Visualization`](label_visualization.md), [`Identify Outliers`](identify_outliers.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Motion Detection`](motion_detection.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Detection Event Log`](detection_event_log.md), [`Camera Calibration`](camera_calibration.md), [`Detections Consensus`](detections_consensus.md), [`SIFT Comparison`](sift_comparison.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`VLM As Detector`](vlm_as_detector.md), [`Detection Offset`](detection_offset.md), [`Identify Changes`](identify_changes.md), [`QR Code Generator`](qr_code_generator.md), [`Slack Notification`](slack_notification.md), [`Mask Area Measurement`](mask_area_measurement.md), [`OCR Model`](ocr_model.md), [`Overlap Filter`](overlap_filter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Corner Visualization`](corner_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Depth Estimation`](depth_estimation.md), [`Velocity`](velocity.md), [`Background Color Visualization`](background_color_visualization.md), [`Email Notification`](email_notification.md), [`Object Detection Model`](object_detection_model.md), [`Image Blur`](image_blur.md)
    - outputs: [`Halo Visualization`](halo_visualization.md), [`Overlap Analysis`](overlap_analysis.md), [`Detections Merge`](detections_merge.md), [`Perspective Correction`](perspective_correction.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Mask Edge Snap`](mask_edge_snap.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Bounding Rectangle`](bounding_rectangle.md), [`Circle Visualization`](circle_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Time in Zone`](timein_zone.md), [`Byte Tracker`](byte_tracker.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Detections Stitch`](detections_stitch.md), [`Path Deviation`](path_deviation.md), [`Florence-2 Model`](florence2_model.md), [`SORT Tracker`](sort_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Line Counter`](line_counter.md), [`Detections Transformation`](detections_transformation.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Time in Zone`](timein_zone.md), [`Camera Focus`](camera_focus.md), [`Distance Measurement`](distance_measurement.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Label Visualization`](label_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Line Counter`](line_counter.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Mask Visualization`](mask_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Blur Visualization`](blur_visualization.md), [`Color Visualization`](color_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Detections Combine`](detections_combine.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Path Deviation`](path_deviation.md), [`Florence-2 Model`](florence2_model.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Detection Event Log`](detection_event_log.md), [`Time in Zone`](timein_zone.md), [`Detections Consensus`](detections_consensus.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Detection Offset`](detection_offset.md), [`Byte Tracker`](byte_tracker.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Trace Visualization`](trace_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Overlap Filter`](overlap_filter.md), [`Crop Visualization`](crop_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Corner Visualization`](corner_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Detections Filter`](detections_filter.md), [`Polygon Visualization`](polygon_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Velocity`](velocity.md), [`Icon Visualization`](icon_visualization.md), [`Size Measurement`](size_measurement.md), [`Background Color Visualization`](background_color_visualization.md), [`Triangle Visualization`](triangle_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`BoT-SORT Tracker` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Input image with embedded video metadata (fps and video_identifier). Used to initialise and retrieve per-video tracker state. When camera motion compensation is enabled, frame pixels are read from this image..
        - `detections` (*Union[[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`rle_instance_segmentation_prediction`](../kinds/rle_instance_segmentation_prediction.md)]*): Detection predictions for the current frame to track..
        - `minimum_iou_threshold_first_assoc` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Minimum fused similarity (IoU × confidence) for the first (high-confidence) association step. Default: 0.2..
        - `minimum_iou_threshold_second_assoc` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Minimum IoU for the second (low-confidence) association step. Default: 0.5..
        - `minimum_iou_threshold_unconfirmed_assoc` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Minimum fused similarity for matching unconfirmed tracks to remaining high-confidence detections. Default: 0.3..
        - `minimum_consecutive_frames` (*[`integer`](../kinds/integer.md)*): Number of consecutive frames a track must be matched before it is emitted as a confirmed track (tracker_id != -1). Default: 2..
        - `lost_track_buffer` (*[`integer`](../kinds/integer.md)*): Number of frames to keep a track alive after it loses its matched detection. Higher values improve occlusion recovery. Default: 30..
        - `track_activation_threshold` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Minimum detection confidence required to spawn a new track. Detections below this threshold are not used to create new tracks. Default: 0.7..
        - `high_conf_det_threshold` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Confidence threshold for high-confidence detections used in association. Default: 0.6..
        - `enable_cmc` (*[`boolean`](../kinds/boolean.md)*): Enable camera motion compensation (uses per-frame image pixels). Recommended for moving cameras..
        - `cmc_downscale` (*[`integer`](../kinds/integer.md)*): Downscale factor applied inside CMC for speed and robustness. Default: 2..
        - `instant_first_frame_activation` (*[`boolean`](../kinds/boolean.md)*): If true, tracks on the first frame receive IDs immediately (paper-style). Default false so new/already-seen outputs match other core trackers..

    - output
    
        - `tracked_detections` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`rle_instance_segmentation_prediction`](../kinds/rle_instance_segmentation_prediction.md)]*): Prediction with detected bounding boxes in form of sv.Detections(...) object if `object_detection_prediction` or Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object if `instance_segmentation_prediction` or Prediction with detected bounding boxes and detected keypoints in form of sv.Detections(...) object if `keypoint_detection_prediction` or Prediction with detected bounding boxes and RLE-encoded segmentation masks in form of sv.Detections(...) object if `rle_instance_segmentation_prediction`.
        - `new_instances` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`rle_instance_segmentation_prediction`](../kinds/rle_instance_segmentation_prediction.md)]*): Prediction with detected bounding boxes in form of sv.Detections(...) object if `object_detection_prediction` or Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object if `instance_segmentation_prediction` or Prediction with detected bounding boxes and detected keypoints in form of sv.Detections(...) object if `keypoint_detection_prediction` or Prediction with detected bounding boxes and RLE-encoded segmentation masks in form of sv.Detections(...) object if `rle_instance_segmentation_prediction`.
        - `already_seen_instances` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`rle_instance_segmentation_prediction`](../kinds/rle_instance_segmentation_prediction.md)]*): Prediction with detected bounding boxes in form of sv.Detections(...) object if `object_detection_prediction` or Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object if `instance_segmentation_prediction` or Prediction with detected bounding boxes and detected keypoints in form of sv.Detections(...) object if `keypoint_detection_prediction` or Prediction with detected bounding boxes and RLE-encoded segmentation masks in form of sv.Detections(...) object if `rle_instance_segmentation_prediction`.



??? tip "Example JSON definition of step `BoT-SORT Tracker` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/trackers_botsort@v1",
	    "image": "<block_does_not_provide_example>",
	    "detections": "$steps.object_detection_model.predictions",
	    "minimum_iou_threshold_first_assoc": 0.2,
	    "minimum_iou_threshold_second_assoc": 0.5,
	    "minimum_iou_threshold_unconfirmed_assoc": 0.3,
	    "minimum_consecutive_frames": 2,
	    "lost_track_buffer": 30,
	    "track_activation_threshold": 0.7,
	    "high_conf_det_threshold": 0.6,
	    "enable_cmc": false,
	    "cmc_method": "sparseOptFlow",
	    "cmc_downscale": 2,
	    "instant_first_frame_activation": false,
	    "instances_cache_size": "<block_does_not_provide_example>"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    


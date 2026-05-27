
# Detections List Roll-Up



??? "Class: `DetectionsListRollUpBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/fusion/detections_list_rollup/v1.py">inference.core.workflows.core_steps.fusion.detections_list_rollup.v1.DetectionsListRollUpBlockV1</a>
    



Rolls up dimensionality from children to parent detections

Useful in scenarios like:
* rolling up results from a secondary model run on crops back to parent images
* rolling up OCR results for dynamically cropped images


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/detections_list_rollup@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `confidence_strategy` | `str` | Strategy to use when merging confidence scores from child detections. Options are 'max', 'mean', or 'min'.. | ✅ |
| `overlap_threshold` | `float` | Minimum overlap ratio (IoU) to consider when merging overlapping detections from child crops. A value of 0.0 merges any overlapping detections, while higher values require greater overlap to merge. Specify between 0.0 and 1.0. A value of 1.0 only merges completely overlapping detections.. | ✅ |
| `keypoint_merge_threshold` | `float` | Keypoint distance (in pixels) to merge keypoint detections if the child detections contain keypoint data.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Detections List Roll-Up` in version `v1`.

    - inputs: [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections Merge`](detections_merge.md), [`Perspective Correction`](perspective_correction.md), [`Anthropic Claude`](anthropic_claude.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Bounding Rectangle`](bounding_rectangle.md), [`Mask Edge Snap`](mask_edge_snap.md), [`SAM 3`](sam3.md), [`Byte Tracker`](byte_tracker.md), [`Time in Zone`](timein_zone.md), [`Byte Tracker`](byte_tracker.md), [`OpenRouter`](open_router.md), [`Detections Stitch`](detections_stitch.md), [`Path Deviation`](path_deviation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen-VL`](qwen_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Template Matching`](template_matching.md), [`SAM 3`](sam3.md), [`Florence-2 Model`](florence2_model.md), [`Object Detection Model`](object_detection_model.md), [`SORT Tracker`](sort_tracker.md), [`SAM 3`](sam3.md), [`Detections Transformation`](detections_transformation.md), [`Line Counter`](line_counter.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Gaze Detection`](gaze_detection.md), [`Google Gemini`](google_gemini.md), [`Google Vision OCR`](google_vision_ocr.md), [`Cosine Similarity`](cosine_similarity.md), [`Object Detection Model`](object_detection_model.md), [`Google Gemini`](google_gemini.md), [`Time in Zone`](timein_zone.md), [`Camera Focus`](camera_focus.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`OpenAI`](open_ai.md), [`Dynamic Zone`](dynamic_zone.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`YOLO-World Model`](yolo_world_model.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`EasyOCR`](easy_ocr.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Gemma API`](google_gemma_api.md), [`Camera Focus`](camera_focus.md), [`Clip Comparison`](clip_comparison.md), [`Identify Outliers`](identify_outliers.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Motion Detection`](motion_detection.md), [`Dimension Collapse`](dimension_collapse.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Detections Combine`](detections_combine.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Image Stack`](image_stack.md), [`Path Deviation`](path_deviation.md), [`Florence-2 Model`](florence2_model.md), [`Detection Event Log`](detection_event_log.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Time in Zone`](timein_zone.md), [`Detections Consensus`](detections_consensus.md), [`Google Gemma`](google_gemma.md), [`Buffer`](buffer.md), [`VLM As Detector`](vlm_as_detector.md), [`VLM As Detector`](vlm_as_detector.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Detection Offset`](detection_offset.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Identify Changes`](identify_changes.md), [`Byte Tracker`](byte_tracker.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Mask Area Measurement`](mask_area_measurement.md), [`OCR Model`](ocr_model.md), [`Overlap Filter`](overlap_filter.md), [`Seg Preview`](seg_preview.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Clip Comparison`](clip_comparison.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Detections Filter`](detections_filter.md), [`Google Gemini`](google_gemini.md), [`Moondream2`](moondream2.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Velocity`](velocity.md), [`Size Measurement`](size_measurement.md), [`Object Detection Model`](object_detection_model.md)
    - outputs: [`Halo Visualization`](halo_visualization.md), [`Grid Visualization`](grid_visualization.md), [`Overlap Analysis`](overlap_analysis.md), [`Perspective Correction`](perspective_correction.md), [`Mask Edge Snap`](mask_edge_snap.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Circle Visualization`](circle_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Time in Zone`](timein_zone.md), [`Byte Tracker`](byte_tracker.md), [`OpenRouter`](open_router.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Detections Stitch`](detections_stitch.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Florence-2 Model`](florence2_model.md), [`Object Detection Model`](object_detection_model.md), [`SORT Tracker`](sort_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemini`](google_gemini.md), [`Camera Focus`](camera_focus.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI`](open_ai.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Halo Visualization`](halo_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Line Counter`](line_counter.md), [`Mask Visualization`](mask_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Color Visualization`](color_visualization.md), [`Detections Combine`](detections_combine.md), [`Line Counter Visualization`](line_counter_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Path Deviation`](path_deviation.md), [`Florence-2 Model`](florence2_model.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Time in Zone`](timein_zone.md), [`Google Gemma`](google_gemma.md), [`Webhook Sink`](webhook_sink.md), [`VLM As Detector`](vlm_as_detector.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Byte Tracker`](byte_tracker.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Email Notification`](email_notification.md), [`Seg Preview`](seg_preview.md), [`Crop Visualization`](crop_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Detections Filter`](detections_filter.md), [`Google Gemini`](google_gemini.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Icon Visualization`](icon_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections Merge`](detections_merge.md), [`Cache Set`](cache_set.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Anthropic Claude`](anthropic_claude.md), [`Bounding Rectangle`](bounding_rectangle.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Path Deviation`](path_deviation.md), [`Qwen-VL`](qwen_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`SAM 3`](sam3.md), [`SAM 3`](sam3.md), [`Line Counter`](line_counter.md), [`Detections Transformation`](detections_transformation.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Time in Zone`](timein_zone.md), [`Distance Measurement`](distance_measurement.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`VLM As Classifier`](vlm_as_classifier.md), [`YOLO-World Model`](yolo_world_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenAI`](open_ai.md), [`Google Gemma API`](google_gemma_api.md), [`Label Visualization`](label_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Motion Detection`](motion_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Detection Event Log`](detection_event_log.md), [`Detections Consensus`](detections_consensus.md), [`Buffer`](buffer.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`VLM As Detector`](vlm_as_detector.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detection Offset`](detection_offset.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI`](open_ai.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Overlap Filter`](overlap_filter.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Corner Visualization`](corner_visualization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Velocity`](velocity.md), [`Size Measurement`](size_measurement.md), [`Background Color Visualization`](background_color_visualization.md), [`Email Notification`](email_notification.md), [`Triangle Visualization`](triangle_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Detections List Roll-Up` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `parent_detection` (*Union[[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md)]*): The parent detection the dimensionality inherits from..
        - `child_detections` (*[`list_of_values`](../kinds/list_of_values.md)*): A list of child detections resulting from higher dimensionality, such as predictions made on dynamic crops. Use the "Dimension Collapse" to  reduce the higher dimensionality result to one that can be used with this. Example: Prediction -> Dimension Collapse -> Detections List Roll-Up.
        - `confidence_strategy` (*[`list_of_values`](../kinds/list_of_values.md)*): Strategy to use when merging confidence scores from child detections. Options are 'max', 'mean', or 'min'..
        - `overlap_threshold` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Minimum overlap ratio (IoU) to consider when merging overlapping detections from child crops. A value of 0.0 merges any overlapping detections, while higher values require greater overlap to merge. Specify between 0.0 and 1.0. A value of 1.0 only merges completely overlapping detections..
        - `keypoint_merge_threshold` (*[`float`](../kinds/float.md)*): Keypoint distance (in pixels) to merge keypoint detections if the child detections contain keypoint data..

    - output
    
        - `rolled_up_detections` (*Union[[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md)]*): Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object if `instance_segmentation_prediction` or Prediction with detected bounding boxes in form of sv.Detections(...) object if `object_detection_prediction` or Prediction with detected bounding boxes and detected keypoints in form of sv.Detections(...) object if `keypoint_detection_prediction`.
        - `crop_zones` ([`list_of_values`](../kinds/list_of_values.md)): List of values of any type.



??? tip "Example JSON definition of step `Detections List Roll-Up` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/detections_list_rollup@v1",
	    "parent_detection": "<block_does_not_provide_example>",
	    "child_detections": "<block_does_not_provide_example>",
	    "confidence_strategy": "min",
	    "overlap_threshold": 0.0,
	    "keypoint_merge_threshold": 0.0
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    


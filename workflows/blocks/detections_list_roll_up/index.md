
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

    - inputs: [`YOLO-World Model`](yolo_world_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Florence-2 Model`](florence2_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Identify Outliers`](identify_outliers.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`OpenAI`](open_ai.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`SAM 3`](sam3.md), [`SAM 3`](sam3.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Google Gemma`](google_gemma.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Motion Detection`](motion_detection.md), [`Time in Zone`](timein_zone.md), [`VLM As Detector`](vlm_as_detector.md), [`Byte Tracker`](byte_tracker.md), [`OpenRouter`](open_router.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Line Counter`](line_counter.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Detections Combine`](detections_combine.md), [`Gaze Detection`](gaze_detection.md), [`Anthropic Claude`](anthropic_claude.md), [`Size Measurement`](size_measurement.md), [`Detections Stabilizer`](detections_stabilizer.md), [`SORT Tracker`](sort_tracker.md), [`Camera Focus`](camera_focus.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dynamic Crop`](dynamic_crop.md), [`Byte Tracker`](byte_tracker.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemma API`](google_gemma_api.md), [`OCR Model`](ocr_model.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Perspective Correction`](perspective_correction.md), [`Identify Changes`](identify_changes.md), [`Seg Preview`](seg_preview.md), [`Moondream2`](moondream2.md), [`Camera Focus`](camera_focus.md), [`Overlap Filter`](overlap_filter.md), [`Anthropic Claude`](anthropic_claude.md), [`Detection Event Log`](detection_event_log.md), [`Image Stack`](image_stack.md), [`Qwen-VL`](qwen_vl.md), [`SAM 3`](sam3.md), [`Detections Stitch`](detections_stitch.md), [`Time in Zone`](timein_zone.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Detections Consensus`](detections_consensus.md), [`Detections Transformation`](detections_transformation.md), [`Detections Merge`](detections_merge.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Detections Filter`](detections_filter.md), [`Path Deviation`](path_deviation.md), [`Google Gemini`](google_gemini.md), [`Google Gemini`](google_gemini.md), [`Object Detection Model`](object_detection_model.md), [`Dimension Collapse`](dimension_collapse.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Object Detection Model`](object_detection_model.md), [`Dynamic Zone`](dynamic_zone.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Vision OCR`](google_vision_ocr.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Buffer`](buffer.md), [`OpenAI`](open_ai.md), [`Velocity`](velocity.md), [`Detection Offset`](detection_offset.md), [`EasyOCR`](easy_ocr.md), [`Path Deviation`](path_deviation.md), [`Clip Comparison`](clip_comparison.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Template Matching`](template_matching.md), [`VLM As Detector`](vlm_as_detector.md), [`Object Detection Model`](object_detection_model.md), [`Google Gemini`](google_gemini.md), [`Cosine Similarity`](cosine_similarity.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Byte Tracker`](byte_tracker.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md)
    - outputs: [`Distance Measurement`](distance_measurement.md), [`OpenAI`](open_ai.md), [`LMM For Classification`](lmm_for_classification.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`SAM 3`](sam3.md), [`Byte Tracker`](byte_tracker.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Halo Visualization`](halo_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Color Visualization`](color_visualization.md), [`OpenRouter`](open_router.md), [`Cache Set`](cache_set.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Line Counter`](line_counter.md), [`VLM As Classifier`](vlm_as_classifier.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Detections Combine`](detections_combine.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`SORT Tracker`](sort_tracker.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Byte Tracker`](byte_tracker.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Halo Visualization`](halo_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Perspective Correction`](perspective_correction.md), [`Trace Visualization`](trace_visualization.md), [`Overlap Filter`](overlap_filter.md), [`Anthropic Claude`](anthropic_claude.md), [`Dot Visualization`](dot_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Background Color Visualization`](background_color_visualization.md), [`Crop Visualization`](crop_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Time in Zone`](timein_zone.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Path Deviation`](path_deviation.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Google Gemini`](google_gemini.md), [`Grid Visualization`](grid_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Velocity`](velocity.md), [`Detection Offset`](detection_offset.md), [`Overlap Analysis`](overlap_analysis.md), [`Clip Comparison`](clip_comparison.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Time in Zone`](timein_zone.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Mask Visualization`](mask_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemini`](google_gemini.md), [`Label Visualization`](label_visualization.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Size Measurement`](size_measurement.md), [`YOLO-World Model`](yolo_world_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Florence-2 Model`](florence2_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SAM 3`](sam3.md), [`Google Gemma`](google_gemma.md), [`Email Notification`](email_notification.md), [`Corner Visualization`](corner_visualization.md), [`Motion Detection`](motion_detection.md), [`Time in Zone`](timein_zone.md), [`VLM As Detector`](vlm_as_detector.md), [`Byte Tracker`](byte_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Email Notification`](email_notification.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Camera Focus`](camera_focus.md), [`Florence-2 Model`](florence2_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Seg Preview`](seg_preview.md), [`Qwen-VL`](qwen_vl.md), [`SAM 3`](sam3.md), [`Detections Stitch`](detections_stitch.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Detections Consensus`](detections_consensus.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Detections Merge`](detections_merge.md), [`Detections Transformation`](detections_transformation.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Detections Filter`](detections_filter.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemini`](google_gemini.md), [`Circle Visualization`](circle_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Object Detection Model`](object_detection_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Line Counter`](line_counter.md), [`Buffer`](buffer.md), [`OpenAI`](open_ai.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Path Deviation`](path_deviation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Webhook Sink`](webhook_sink.md), [`Object Detection Model`](object_detection_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Blur Visualization`](blur_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Detections List Roll-Up` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `parent_detection` (*Union[[`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): The parent detection the dimensionality inherits from..
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


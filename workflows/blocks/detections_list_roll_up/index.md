
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

    - inputs: [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Google Gemma API`](google_gemma_api.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Path Deviation`](path_deviation.md), [`Image Stack`](image_stack.md), [`Detection Offset`](detection_offset.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Detection Event Log`](detection_event_log.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Detections Transformation`](detections_transformation.md), [`Perspective Correction`](perspective_correction.md), [`Clip Comparison`](clip_comparison.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Google Vision OCR`](google_vision_ocr.md), [`Detections Merge`](detections_merge.md), [`Dynamic Zone`](dynamic_zone.md), [`Detections Combine`](detections_combine.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Qwen-VL`](qwen_vl.md), [`Camera Focus`](camera_focus.md), [`Size Measurement`](size_measurement.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Object Detection Model`](object_detection_model.md), [`Bounding Rectangle`](bounding_rectangle.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Google Gemma`](google_gemma.md), [`Byte Tracker`](byte_tracker.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Velocity`](velocity.md), [`Identify Outliers`](identify_outliers.md), [`Byte Tracker`](byte_tracker.md), [`Identify Changes`](identify_changes.md), [`Dynamic Crop`](dynamic_crop.md), [`Detections Stitch`](detections_stitch.md), [`Path Deviation`](path_deviation.md), [`Template Matching`](template_matching.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Camera Focus`](camera_focus.md), [`Gaze Detection`](gaze_detection.md), [`OpenRouter`](open_router.md), [`OpenAI`](open_ai.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`VLM As Detector`](vlm_as_detector.md), [`Florence-2 Model`](florence2_model.md), [`Motion Detection`](motion_detection.md), [`OCR Model`](ocr_model.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Detections Filter`](detections_filter.md), [`Object Detection Model`](object_detection_model.md), [`Dimension Collapse`](dimension_collapse.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`YOLO-World Model`](yolo_world_model.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Google Gemini`](google_gemini.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemini`](google_gemini.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Detections Consensus`](detections_consensus.md), [`Buffer`](buffer.md), [`Byte Tracker`](byte_tracker.md), [`SAM 3`](sam3.md), [`Florence-2 Model`](florence2_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Mask Edge Snap`](mask_edge_snap.md), [`OpenAI`](open_ai.md), [`Moondream2`](moondream2.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemini`](google_gemini.md), [`EasyOCR`](easy_ocr.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Overlap Filter`](overlap_filter.md), [`SAM 3`](sam3.md), [`Time in Zone`](timein_zone.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`SORT Tracker`](sort_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Track Class Lock`](track_class_lock.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Object Detection Model`](object_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Time in Zone`](timein_zone.md), [`Cosine Similarity`](cosine_similarity.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`SAM 3`](sam3.md), [`Seg Preview`](seg_preview.md)
    - outputs: [`VLM As Classifier`](vlm_as_classifier.md), [`Line Counter`](line_counter.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Path Deviation`](path_deviation.md), [`Trace Visualization`](trace_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Icon Visualization`](icon_visualization.md), [`Color Visualization`](color_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Perspective Correction`](perspective_correction.md), [`Corner Visualization`](corner_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Detections Merge`](detections_merge.md), [`Halo Visualization`](halo_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Qwen-VL`](qwen_vl.md), [`Email Notification`](email_notification.md), [`Halo Visualization`](halo_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Google Gemma`](google_gemma.md), [`Background Color Visualization`](background_color_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Email Notification`](email_notification.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Polygon Visualization`](polygon_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`OpenRouter`](open_router.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`VLM As Detector`](vlm_as_detector.md), [`OpenAI`](open_ai.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Motion Detection`](motion_detection.md), [`Detections Filter`](detections_filter.md), [`Blur Visualization`](blur_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemini`](google_gemini.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Buffer`](buffer.md), [`Webhook Sink`](webhook_sink.md), [`Byte Tracker`](byte_tracker.md), [`Florence-2 Model`](florence2_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Mask Edge Snap`](mask_edge_snap.md), [`OpenAI`](open_ai.md), [`Line Counter`](line_counter.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemini`](google_gemini.md), [`Triangle Visualization`](triangle_visualization.md), [`Overlap Filter`](overlap_filter.md), [`Time in Zone`](timein_zone.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Time in Zone`](timein_zone.md), [`Google Gemma API`](google_gemma_api.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Distance Measurement`](distance_measurement.md), [`Detection Offset`](detection_offset.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Detections Transformation`](detections_transformation.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Detections Combine`](detections_combine.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Size Measurement`](size_measurement.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Classification Label Visualization`](classification_label_visualization.md), [`Bounding Rectangle`](bounding_rectangle.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Event Writer`](event_writer.md), [`Grid Visualization`](grid_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Velocity`](velocity.md), [`Label Visualization`](label_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Dot Visualization`](dot_visualization.md), [`Cache Set`](cache_set.md), [`Dynamic Crop`](dynamic_crop.md), [`Path Deviation`](path_deviation.md), [`Circle Visualization`](circle_visualization.md), [`Detections Stitch`](detections_stitch.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Camera Focus`](camera_focus.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Overlap Analysis`](overlap_analysis.md), [`Object Detection Model`](object_detection_model.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Detections Consensus`](detections_consensus.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`SAM 3`](sam3.md), [`PLC Reader`](plc_reader.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SAM 3`](sam3.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SORT Tracker`](sort_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Track Class Lock`](track_class_lock.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Object Detection Model`](object_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Time in Zone`](timein_zone.md), [`Polygon Visualization`](polygon_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`SAM 3`](sam3.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Seg Preview`](seg_preview.md)

    
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


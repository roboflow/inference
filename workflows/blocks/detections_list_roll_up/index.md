
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

    - inputs: [`YOLO-World Model`](yolo_world_model.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`SAM 3`](sam3.md), [`Dynamic Zone`](dynamic_zone.md), [`Florence-2 Model`](florence2_model.md), [`Byte Tracker`](byte_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`SORT Tracker`](sort_tracker.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Clip Comparison`](clip_comparison.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Detections Consensus`](detections_consensus.md), [`Time in Zone`](timein_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`OCR Model`](ocr_model.md), [`Cosine Similarity`](cosine_similarity.md), [`Byte Tracker`](byte_tracker.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Google Gemini`](google_gemini.md), [`OpenAI`](open_ai.md), [`Camera Focus`](camera_focus.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Time in Zone`](timein_zone.md), [`Track Class Lock`](track_class_lock.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Velocity`](velocity.md), [`Buffer`](buffer.md), [`Identify Outliers`](identify_outliers.md), [`Google Gemma API`](google_gemma_api.md), [`SAM 3`](sam3.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Identify Changes`](identify_changes.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`VLM As Detector`](vlm_as_detector.md), [`Path Deviation`](path_deviation.md), [`Moondream2`](moondream2.md), [`Dimension Collapse`](dimension_collapse.md), [`Motion Detection`](motion_detection.md), [`OpenRouter`](open_router.md), [`SAM 3`](sam3.md), [`Detection Event Log`](detection_event_log.md), [`Anthropic Claude`](anthropic_claude.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Detections Combine`](detections_combine.md), [`Byte Tracker`](byte_tracker.md), [`Google Vision OCR`](google_vision_ocr.md), [`Line Counter`](line_counter.md), [`Size Measurement`](size_measurement.md), [`Overlap Filter`](overlap_filter.md), [`Anthropic Claude`](anthropic_claude.md), [`EasyOCR`](easy_ocr.md), [`Template Matching`](template_matching.md), [`Google Gemini`](google_gemini.md), [`Clip Comparison`](clip_comparison.md), [`Detections Stitch`](detections_stitch.md), [`Seg Preview`](seg_preview.md), [`Google Gemma`](google_gemma.md), [`Google Gemini`](google_gemini.md), [`OpenAI`](open_ai.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Object Detection Model`](object_detection_model.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Image Stack`](image_stack.md), [`Gaze Detection`](gaze_detection.md), [`Detections Merge`](detections_merge.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Path Deviation`](path_deviation.md), [`Detection Offset`](detection_offset.md), [`VLM As Detector`](vlm_as_detector.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Camera Focus`](camera_focus.md), [`Perspective Correction`](perspective_correction.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Dynamic Crop`](dynamic_crop.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Florence-2 Model`](florence2_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Detections Filter`](detections_filter.md), [`OpenAI`](open_ai.md), [`Detections Transformation`](detections_transformation.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Qwen-VL`](qwen_vl.md)
    - outputs: [`YOLO-World Model`](yolo_world_model.md), [`SAM 3`](sam3.md), [`Dynamic Zone`](dynamic_zone.md), [`Byte Tracker`](byte_tracker.md), [`SORT Tracker`](sort_tracker.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Pixelate Visualization`](pixelate_visualization.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Velocity`](velocity.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Corner Visualization`](corner_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`PLC Reader`](plc_reader.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`VLM As Detector`](vlm_as_detector.md), [`Motion Detection`](motion_detection.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Size Measurement`](size_measurement.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Gemini`](google_gemini.md), [`VLM As Classifier`](vlm_as_classifier.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Circle Visualization`](circle_visualization.md), [`OpenAI`](open_ai.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Path Deviation`](path_deviation.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`VLM As Detector`](vlm_as_detector.md), [`Mask Visualization`](mask_visualization.md), [`Dot Visualization`](dot_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Perspective Correction`](perspective_correction.md), [`Overlap Analysis`](overlap_analysis.md), [`Blur Visualization`](blur_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Dynamic Crop`](dynamic_crop.md), [`Trace Visualization`](trace_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Email Notification`](email_notification.md), [`Detections Transformation`](detections_transformation.md), [`Background Color Visualization`](background_color_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Florence-2 Model`](florence2_model.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Clip Comparison`](clip_comparison.md), [`Cache Set`](cache_set.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Detections Consensus`](detections_consensus.md), [`Time in Zone`](timein_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`Byte Tracker`](byte_tracker.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemini`](google_gemini.md), [`Email Notification`](email_notification.md), [`Crop Visualization`](crop_visualization.md), [`Time in Zone`](timein_zone.md), [`Label Visualization`](label_visualization.md), [`Track Class Lock`](track_class_lock.md), [`Buffer`](buffer.md), [`Distance Measurement`](distance_measurement.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Triangle Visualization`](triangle_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`SAM 3`](sam3.md), [`Halo Visualization`](halo_visualization.md), [`Path Deviation`](path_deviation.md), [`OpenRouter`](open_router.md), [`Detection Event Log`](detection_event_log.md), [`Detections Combine`](detections_combine.md), [`Byte Tracker`](byte_tracker.md), [`Line Counter`](line_counter.md), [`Overlap Filter`](overlap_filter.md), [`Grid Visualization`](grid_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Detections Stitch`](detections_stitch.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Seg Preview`](seg_preview.md), [`Google Gemma`](google_gemma.md), [`Line Counter`](line_counter.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Object Detection Model`](object_detection_model.md), [`Event Writer`](event_writer.md), [`SAM 3 Interactive`](sam3_interactive.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Detections Merge`](detections_merge.md), [`Detection Offset`](detection_offset.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Camera Focus`](camera_focus.md), [`Color Visualization`](color_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Webhook Sink`](webhook_sink.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Icon Visualization`](icon_visualization.md), [`Detections Filter`](detections_filter.md), [`OpenAI`](open_ai.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Ellipse Visualization`](ellipse_visualization.md), [`SAM2 Video Tracker`](sam2_video_tracker.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Detections List Roll-Up` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `parent_detection` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): The parent detection the dimensionality inherits from..
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


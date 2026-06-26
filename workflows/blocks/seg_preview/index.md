
# Seg Preview



??? "Class: `SegPreviewBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/seg_preview/v1.py">inference.core.workflows.core_steps.models.foundation.seg_preview.v1.SegPreviewBlockV1</a>
    


Seg Preview

### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/seg-preview@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `class_names` | `Union[List[str], str]` | List of classes to recognise. | ✅ |
| `threshold` | `float` | Threshold for predicted mask scores. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-cloud-off-outline:{ style="color: #546e7a" } `requires_internet` — air-gapped / offline deployments
:   This block depends on a service that is not reachable from fully offline / air-gapped deployments.

:material-shield-alert:{ style="color: #d32f2f" } `hard` — runtime `self_hosted_cpu`, `self_hosted_gpu`, `inference_pipeline`
:   Seg Preview calls the Roboflow-internal API_BASE_URL/inferenceproxy/seg-preview endpoint, which is only reachable from Roboflow-hosted runtimes (HOSTED_SERVERLESS, DEDICATED_DEPLOYMENT). Self-hosted deployments cannot run this block.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Seg Preview` in version `v1`.

    - inputs: [`VLM As Classifier`](vlm_as_classifier.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Trace Visualization`](trace_visualization.md), [`Image Stack`](image_stack.md), [`Anthropic Claude`](anthropic_claude.md), [`Icon Visualization`](icon_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Color Visualization`](color_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Perspective Correction`](perspective_correction.md), [`Corner Visualization`](corner_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Halo Visualization`](halo_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Qwen-VL`](qwen_vl.md), [`Email Notification`](email_notification.md), [`Halo Visualization`](halo_visualization.md), [`Google Gemma`](google_gemma.md), [`Background Color Visualization`](background_color_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Email Notification`](email_notification.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Text Display`](text_display.md), [`Polygon Visualization`](polygon_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Image Preprocessing`](image_preprocessing.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Relative Static Crop`](relative_static_crop.md), [`OpenRouter`](open_router.md), [`OpenAI`](open_ai.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Motion Detection`](motion_detection.md), [`OCR Model`](ocr_model.md), [`Blur Visualization`](blur_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`Depth Estimation`](depth_estimation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Gemini`](google_gemini.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemini`](google_gemini.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Background Subtraction`](background_subtraction.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Buffer`](buffer.md), [`CSV Formatter`](csv_formatter.md), [`Webhook Sink`](webhook_sink.md), [`Stitch Images`](stitch_images.md), [`Florence-2 Model`](florence2_model.md), [`Current Time`](current_time.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Contrast Equalization`](contrast_equalization.md), [`OpenAI`](open_ai.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemini`](google_gemini.md), [`Triangle Visualization`](triangle_visualization.md), [`Slack Notification`](slack_notification.md), [`SIFT`](sift.md), [`Local File Sink`](local_file_sink.md), [`Cosine Similarity`](cosine_similarity.md), [`Image Contours`](image_contours.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`GLM-OCR`](glmocr.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Image Slicer`](image_slicer.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Google Gemma API`](google_gemma_api.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Image Threshold`](image_threshold.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Camera Calibration`](camera_calibration.md), [`QR Code Generator`](qr_code_generator.md), [`S3 Sink`](s3_sink.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Google Vision OCR`](google_vision_ocr.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Image Blur`](image_blur.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Focus`](camera_focus.md), [`Size Measurement`](size_measurement.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Event Writer`](event_writer.md), [`Grid Visualization`](grid_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Mask Visualization`](mask_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Image Slicer`](image_slicer.md), [`Label Visualization`](label_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Dot Visualization`](dot_visualization.md), [`Identify Changes`](identify_changes.md), [`Dynamic Crop`](dynamic_crop.md), [`Circle Visualization`](circle_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Camera Focus`](camera_focus.md), [`Gaze Detection`](gaze_detection.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`CogVLM`](cog_vlm.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`LMM`](lmm.md), [`OpenAI`](open_ai.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`EasyOCR`](easy_ocr.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`PLC Writer`](plc_writer.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Anthropic Claude`](anthropic_claude.md), [`Object Detection Model`](object_detection_model.md), [`MQTT Writer`](mqtt_writer.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md)
    - outputs: [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Path Deviation`](path_deviation.md), [`Trace Visualization`](trace_visualization.md), [`Distance Measurement`](distance_measurement.md), [`Detection Offset`](detection_offset.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Icon Visualization`](icon_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Color Visualization`](color_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Corner Visualization`](corner_visualization.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Detections Merge`](detections_merge.md), [`Halo Visualization`](halo_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Detections Combine`](detections_combine.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Size Measurement`](size_measurement.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Bounding Rectangle`](bounding_rectangle.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Event Writer`](event_writer.md), [`Mask Visualization`](mask_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Background Color Visualization`](background_color_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Velocity`](velocity.md), [`Label Visualization`](label_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Dot Visualization`](dot_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Path Deviation`](path_deviation.md), [`Circle Visualization`](circle_visualization.md), [`Detections Stitch`](detections_stitch.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Camera Focus`](camera_focus.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Florence-2 Model`](florence2_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Detections Filter`](detections_filter.md), [`Overlap Analysis`](overlap_analysis.md), [`Blur Visualization`](blur_visualization.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Detections Consensus`](detections_consensus.md), [`Byte Tracker`](byte_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Line Counter`](line_counter.md), [`Triangle Visualization`](triangle_visualization.md), [`Overlap Filter`](overlap_filter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Time in Zone`](timein_zone.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Detections Stabilizer`](detections_stabilizer.md), [`SORT Tracker`](sort_tracker.md), [`Track Class Lock`](track_class_lock.md), [`Time in Zone`](timein_zone.md), [`Polygon Visualization`](polygon_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Model Comparison Visualization`](model_comparison_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Seg Preview` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `class_names` (*Union[[`list_of_values`](../kinds/list_of_values.md), [`string`](../kinds/string.md)]*): List of classes to recognise.
        - `threshold` (*[`float`](../kinds/float.md)*): Threshold for predicted mask scores.

    - output
    
        - `predictions` ([`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)): Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object.



??? tip "Example JSON definition of step `Seg Preview` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/seg-preview@v1",
	    "images": "$inputs.image",
	    "class_names": [
	        "car",
	        "person"
	    ],
	    "threshold": 0.3
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    


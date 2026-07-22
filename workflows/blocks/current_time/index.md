
# Current Time



??? "Class: `CurrentTimeBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/formatters/current_time/v1.py">inference.core.workflows.core_steps.formatters.current_time.v1.CurrentTimeBlockV1</a>
    



Output the current date and time for a given timezone.

Provide one of the curated timezone options (for example `America/New_York`,
`Europe/Berlin`, or `UTC`) and the block returns the current moment in that
timezone. The block produces a `timestamp` (a timezone-aware `datetime` object you
can pass to other blocks), along with ready-to-use `iso_string`, `date`, and
`time` strings.

The timezone may be a literal value typed into the block, or a reference to a workflow
input or another step's output.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/current_time@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `timezone` | `str` | Curated IANA timezone name to report the current time in.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Current Time` in version `v1`.

    - inputs: [`Google Gemini`](google_gemini.md), [`OpenAI`](open_ai.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Anthropic Claude`](anthropic_claude.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Webhook Sink`](webhook_sink.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI`](open_ai.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Google Gemini`](google_gemini.md), [`Email Notification`](email_notification.md), [`EasyOCR`](easy_ocr.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Florence-2 Model`](florence2_model.md), [`LMM`](lmm.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`OCR Model`](ocr_model.md), [`Slack Notification`](slack_notification.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Clip Comparison`](clip_comparison.md), [`Email Notification`](email_notification.md), [`VLM As Detector`](vlm_as_detector.md), [`Cosmos 3`](cosmos3.md), [`CogVLM`](cog_vlm.md), [`Qwen-VL`](qwen_vl.md), [`PP-OCR`](ppocr.md), [`Google Gemma`](google_gemma.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`PLC Writer`](plc_writer.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemma API`](google_gemma_api.md), [`OpenAI`](open_ai.md), [`CSV Formatter`](csv_formatter.md), [`GLM-OCR`](glmocr.md), [`Current Time`](current_time.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Anthropic Claude`](anthropic_claude.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Florence-2 Model`](florence2_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Google Vision OCR`](google_vision_ocr.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`MQTT Writer`](mqtt_writer.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Local File Sink`](local_file_sink.md), [`OpenRouter`](open_router.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`OpenAI`](open_ai.md), [`Google Gemini`](google_gemini.md), [`S3 Sink`](s3_sink.md), [`Event Writer`](event_writer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md)
    - outputs: [`SAM 3`](sam3.md), [`Image Preprocessing`](image_preprocessing.md), [`Anthropic Claude`](anthropic_claude.md), [`Time in Zone`](timein_zone.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Cache Get`](cache_get.md), [`Path Deviation`](path_deviation.md), [`SIFT Comparison`](sift_comparison.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Email Notification`](email_notification.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Distance Measurement`](distance_measurement.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Florence-2 Model`](florence2_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`YOLO-World Model`](yolo_world_model.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Slack Notification`](slack_notification.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Clip Comparison`](clip_comparison.md), [`CogVLM`](cog_vlm.md), [`Corner Visualization`](corner_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Color Visualization`](color_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OpenAI`](open_ai.md), [`Triangle Visualization`](triangle_visualization.md), [`Time in Zone`](timein_zone.md), [`Object Detection Model`](object_detection_model.md), [`SAM 3`](sam3.md), [`Image Threshold`](image_threshold.md), [`SAM 3`](sam3.md), [`Current Time`](current_time.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`QR Code Generator`](qr_code_generator.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Florence-2 Model`](florence2_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Path Deviation`](path_deviation.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Moondream2`](moondream2.md), [`Label Visualization`](label_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Line Counter`](line_counter.md), [`MQTT Writer`](mqtt_writer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Local File Sink`](local_file_sink.md), [`Cache Set`](cache_set.md), [`Event Writer`](event_writer.md), [`Google Gemini`](google_gemini.md), [`Depth Estimation`](depth_estimation.md), [`Seg Preview`](seg_preview.md), [`Google Gemini`](google_gemini.md), [`OpenAI`](open_ai.md), [`Trace Visualization`](trace_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Pixel Color Count`](pixel_color_count.md), [`LMM For Classification`](lmm_for_classification.md), [`Webhook Sink`](webhook_sink.md), [`Halo Visualization`](halo_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dot Visualization`](dot_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Google Gemini`](google_gemini.md), [`Keypoint Visualization`](keypoint_visualization.md), [`LMM`](lmm.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Time in Zone`](timein_zone.md), [`Circle Visualization`](circle_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Email Notification`](email_notification.md), [`Halo Visualization`](halo_visualization.md), [`Cosmos 3`](cosmos3.md), [`Polygon Visualization`](polygon_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Google Gemma`](google_gemma.md), [`Crop Visualization`](crop_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Size Measurement`](size_measurement.md), [`Icon Visualization`](icon_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Google Gemma API`](google_gemma_api.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Image Blur`](image_blur.md), [`Background Color Visualization`](background_color_visualization.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Detections Stitch`](detections_stitch.md), [`GLM-OCR`](glmocr.md), [`Anthropic Claude`](anthropic_claude.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`Perspective Correction`](perspective_correction.md), [`Polygon Visualization`](polygon_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Text Display`](text_display.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenRouter`](open_router.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`OpenAI`](open_ai.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`S3 Sink`](s3_sink.md), [`Line Counter`](line_counter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Current Time` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `timezone` (*[`string`](../kinds/string.md)*): Curated IANA timezone name to report the current time in..

    - output
    
        - `timestamp` ([`timestamp`](../kinds/timestamp.md)): Timestamp object.
        - `iso_string` ([`string`](../kinds/string.md)): String value.
        - `date` ([`string`](../kinds/string.md)): String value.
        - `time` ([`string`](../kinds/string.md)): String value.



??? tip "Example JSON definition of step `Current Time` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/current_time@v1",
	    "timezone": "UTC"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    


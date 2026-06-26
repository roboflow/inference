
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

    - inputs: [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Google Gemma API`](google_gemma_api.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`CogVLM`](cog_vlm.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Gemini`](google_gemini.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Google Gemini`](google_gemini.md), [`S3 Sink`](s3_sink.md), [`CSV Formatter`](csv_formatter.md), [`LMM For Classification`](lmm_for_classification.md), [`Webhook Sink`](webhook_sink.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`LMM`](lmm.md), [`OpenAI`](open_ai.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Google Vision OCR`](google_vision_ocr.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Florence-2 Model`](florence2_model.md), [`Qwen-VL`](qwen_vl.md), [`Current Time`](current_time.md), [`Email Notification`](email_notification.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`OpenAI`](open_ai.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemini`](google_gemini.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Slack Notification`](slack_notification.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemma`](google_gemma.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`EasyOCR`](easy_ocr.md), [`Event Writer`](event_writer.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`PLC Writer`](plc_writer.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Anthropic Claude`](anthropic_claude.md), [`Object Detection Model`](object_detection_model.md), [`Local File Sink`](local_file_sink.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`MQTT Writer`](mqtt_writer.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`GLM-OCR`](glmocr.md), [`OpenRouter`](open_router.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`OCR Model`](ocr_model.md)
    - outputs: [`Line Counter`](line_counter.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Trace Visualization`](trace_visualization.md), [`Path Deviation`](path_deviation.md), [`Anthropic Claude`](anthropic_claude.md), [`Icon Visualization`](icon_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Color Visualization`](color_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Perspective Correction`](perspective_correction.md), [`Corner Visualization`](corner_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Halo Visualization`](halo_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Email Notification`](email_notification.md), [`Halo Visualization`](halo_visualization.md), [`Google Gemma`](google_gemma.md), [`Background Color Visualization`](background_color_visualization.md), [`Email Notification`](email_notification.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Text Display`](text_display.md), [`Polygon Visualization`](polygon_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`OpenRouter`](open_router.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Depth Estimation`](depth_estimation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemini`](google_gemini.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Florence-2 Model`](florence2_model.md), [`Current Time`](current_time.md), [`Contrast Equalization`](contrast_equalization.md), [`OpenAI`](open_ai.md), [`Moondream2`](moondream2.md), [`Line Counter`](line_counter.md), [`Google Gemini`](google_gemini.md), [`Triangle Visualization`](triangle_visualization.md), [`Slack Notification`](slack_notification.md), [`Time in Zone`](timein_zone.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Local File Sink`](local_file_sink.md), [`Pixel Color Count`](pixel_color_count.md), [`GLM-OCR`](glmocr.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Time in Zone`](timein_zone.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Distance Measurement`](distance_measurement.md), [`Image Threshold`](image_threshold.md), [`QR Code Generator`](qr_code_generator.md), [`S3 Sink`](s3_sink.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Google Vision OCR`](google_vision_ocr.md), [`Image Blur`](image_blur.md), [`Morphological Transformation`](morphological_transformation.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Size Measurement`](size_measurement.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Event Writer`](event_writer.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Mask Visualization`](mask_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Label Visualization`](label_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Dot Visualization`](dot_visualization.md), [`Cache Set`](cache_set.md), [`Dynamic Crop`](dynamic_crop.md), [`Detections Stitch`](detections_stitch.md), [`Circle Visualization`](circle_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Path Deviation`](path_deviation.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`CogVLM`](cog_vlm.md), [`Object Detection Model`](object_detection_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`LMM`](lmm.md), [`OpenAI`](open_ai.md), [`SAM 3`](sam3.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SAM 3`](sam3.md), [`Cache Get`](cache_get.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Anthropic Claude`](anthropic_claude.md), [`Time in Zone`](timein_zone.md), [`MQTT Writer`](mqtt_writer.md), [`Polygon Visualization`](polygon_visualization.md), [`SAM 3`](sam3.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Seg Preview`](seg_preview.md)

    
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


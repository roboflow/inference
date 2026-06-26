
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

    - inputs: [`Llama 3.2 Vision`](llama3.2_vision.md), [`Google Vision OCR`](google_vision_ocr.md), [`Google Gemini`](google_gemini.md), [`Anthropic Claude`](anthropic_claude.md), [`VLM As Classifier`](vlm_as_classifier.md), [`EasyOCR`](easy_ocr.md), [`CogVLM`](cog_vlm.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`OpenAI`](open_ai.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI`](open_ai.md), [`Object Detection Model`](object_detection_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Qwen-VL`](qwen_vl.md), [`Slack Notification`](slack_notification.md), [`Email Notification`](email_notification.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`OpenRouter`](open_router.md), [`CSV Formatter`](csv_formatter.md), [`Current Time`](current_time.md), [`Google Gemma`](google_gemma.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`VLM As Detector`](vlm_as_detector.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenAI`](open_ai.md), [`Local File Sink`](local_file_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`LMM For Classification`](lmm_for_classification.md), [`Anthropic Claude`](anthropic_claude.md), [`Webhook Sink`](webhook_sink.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`OCR Model`](ocr_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Email Notification`](email_notification.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`S3 Sink`](s3_sink.md), [`GLM-OCR`](glmocr.md), [`LMM`](lmm.md), [`Florence-2 Model`](florence2_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Clip Comparison`](clip_comparison.md), [`MQTT Writer`](mqtt_writer.md), [`PLC Writer`](plc_writer.md), [`Event Writer`](event_writer.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemma API`](google_gemma_api.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Stitch OCR Detections`](stitch_ocr_detections.md)
    - outputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Threshold`](image_threshold.md), [`Triangle Visualization`](triangle_visualization.md), [`OpenAI`](open_ai.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Qwen-VL`](qwen_vl.md), [`Slack Notification`](slack_notification.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Florence-2 Model`](florence2_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Halo Visualization`](halo_visualization.md), [`Webhook Sink`](webhook_sink.md), [`SAM 3`](sam3.md), [`SAM 3`](sam3.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`QR Code Generator`](qr_code_generator.md), [`Google Gemini`](google_gemini.md), [`YOLO-World Model`](yolo_world_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Label Visualization`](label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Halo Visualization`](halo_visualization.md), [`Path Deviation`](path_deviation.md), [`Florence-2 Model`](florence2_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Morphological Transformation`](morphological_transformation.md), [`Text Display`](text_display.md), [`MQTT Writer`](mqtt_writer.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Google Gemini`](google_gemini.md), [`Google Gemma API`](google_gemma_api.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Polygon Visualization`](polygon_visualization.md), [`Moondream2`](moondream2.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Google Gemini`](google_gemini.md), [`Time in Zone`](timein_zone.md), [`Time in Zone`](timein_zone.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Dot Visualization`](dot_visualization.md), [`Local File Sink`](local_file_sink.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Email Notification`](email_notification.md), [`Line Counter`](line_counter.md), [`Reference Path Visualization`](reference_path_visualization.md), [`S3 Sink`](s3_sink.md), [`LMM`](lmm.md), [`Distance Measurement`](distance_measurement.md), [`Cache Set`](cache_set.md), [`Size Measurement`](size_measurement.md), [`Trace Visualization`](trace_visualization.md), [`SAM 3`](sam3.md), [`Circle Visualization`](circle_visualization.md), [`Time in Zone`](timein_zone.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Event Writer`](event_writer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Path Deviation`](path_deviation.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Blur`](image_blur.md), [`Google Vision OCR`](google_vision_ocr.md), [`Morphological Transformation`](morphological_transformation.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Email Notification`](email_notification.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`OpenRouter`](open_router.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`OpenAI`](open_ai.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Mask Visualization`](mask_visualization.md), [`OpenAI`](open_ai.md), [`LMM For Classification`](lmm_for_classification.md), [`Anthropic Claude`](anthropic_claude.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Anthropic Claude`](anthropic_claude.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Object Detection Model`](object_detection_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Cache Get`](cache_get.md), [`Seg Preview`](seg_preview.md), [`Clip Comparison`](clip_comparison.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`CogVLM`](cog_vlm.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Corner Visualization`](corner_visualization.md), [`OpenAI`](open_ai.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Depth Estimation`](depth_estimation.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Line Counter`](line_counter.md), [`Current Time`](current_time.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Google Gemma`](google_gemma.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Icon Visualization`](icon_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections Stitch`](detections_stitch.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Perspective Correction`](perspective_correction.md), [`Crop Visualization`](crop_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`GLM-OCR`](glmocr.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Pixel Color Count`](pixel_color_count.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md)

    
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


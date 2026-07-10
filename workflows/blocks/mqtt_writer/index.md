
# MQTT Writer



??? "Class: `MQTTWriterSinkBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/enterprise/workflows/enterprise_blocks/sinks/mqtt_writer/v1.py">inference.enterprise.workflows.enterprise_blocks.sinks.mqtt_writer.v1.MQTTWriterSinkBlockV1</a>
    



MQTT Writer block for publishing messages to an MQTT broker.

This block is blocking on connect and publish operations.

Outputs:
    - error_status (bool): Indicates if an error occurred during the MQTT publishing process.
                          True if there was an error, False if successful.
    - message (str): Status message describing the result of the operation.
                    Contains error details if error_status is True,
                    or success confirmation if error_status is False.


### Type identifier

Use the following identifier in step `"type"` field: `mqtt_writer_sink@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `host` | `str` | Host of the MQTT broker.. | ✅ |
| `port` | `int` | Port of the MQTT broker.. | ✅ |
| `topic` | `str` | MQTT topic to publish the message to.. | ✅ |
| `message` | `str` | Message to be published.. | ✅ |
| `qos` | `int` | Quality of Service level for the message.. | ✅ |
| `retain` | `bool` | Whether the message should be retained by the broker.. | ✅ |
| `timeout` | `float` | Timeout for connecting to the MQTT broker and for sending MQTT messages.. | ✅ |
| `username` | `str` | Username for MQTT broker authentication.. | ✅ |
| `password` | `str` | Password for MQTT broker authentication.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `MQTT Writer` in version `v1`.

    - inputs: [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`SIFT Comparison`](sift_comparison.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`PLC Writer`](plc_writer.md), [`JSON Parser`](json_parser.md), [`Distance Measurement`](distance_measurement.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`CogVLM`](cog_vlm.md), [`MQTT Writer`](mqtt_writer.md), [`Google Gemini`](google_gemini.md), [`OCR Model`](ocr_model.md), [`GLM-OCR`](glmocr.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemma API`](google_gemma_api.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`VLM As Detector`](vlm_as_detector.md), [`Florence-2 Model`](florence2_model.md), [`LMM`](lmm.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`Line Counter`](line_counter.md), [`LMM For Classification`](lmm_for_classification.md), [`Event Writer`](event_writer.md), [`OpenAI`](open_ai.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Google Gemma`](google_gemma.md), [`Slack Notification`](slack_notification.md), [`Email Notification`](email_notification.md), [`Perspective Correction`](perspective_correction.md), [`Camera Focus`](camera_focus.md), [`Detections Consensus`](detections_consensus.md), [`Line Counter`](line_counter.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Object Detection Model`](object_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Google Gemini`](google_gemini.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Local File Sink`](local_file_sink.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`PP-OCR`](ppocr.md), [`EasyOCR`](easy_ocr.md), [`Anthropic Claude`](anthropic_claude.md), [`CSV Formatter`](csv_formatter.md), [`Dynamic Zone`](dynamic_zone.md), [`Motion Detection`](motion_detection.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Current Time`](current_time.md), [`PLC Reader`](plc_reader.md), [`Camera Focus`](camera_focus.md), [`Google Gemini`](google_gemini.md), [`Image Stack`](image_stack.md), [`Qwen-VL`](qwen_vl.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Gaze Detection`](gaze_detection.md), [`Google Vision OCR`](google_vision_ocr.md), [`Webhook Sink`](webhook_sink.md), [`OpenAI`](open_ai.md), [`VLM As Detector`](vlm_as_detector.md), [`Identify Outliers`](identify_outliers.md), [`Cosine Similarity`](cosine_similarity.md), [`S3 Sink`](s3_sink.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Detection Event Log`](detection_event_log.md), [`Template Matching`](template_matching.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Contours`](image_contours.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Clip Comparison`](clip_comparison.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`OpenRouter`](open_router.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Identify Changes`](identify_changes.md), [`Email Notification`](email_notification.md), [`OpenAI`](open_ai.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Pixel Color Count`](pixel_color_count.md), [`SIFT Comparison`](sift_comparison.md)
    - outputs: [`Keypoint Visualization`](keypoint_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Object Detection Model`](object_detection_model.md), [`SAM 3`](sam3.md), [`PLC Writer`](plc_writer.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Distance Measurement`](distance_measurement.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Image Threshold`](image_threshold.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`CogVLM`](cog_vlm.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`MQTT Writer`](mqtt_writer.md), [`Contrast Equalization`](contrast_equalization.md), [`Google Gemini`](google_gemini.md), [`GLM-OCR`](glmocr.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemma API`](google_gemma_api.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Florence-2 Model`](florence2_model.md), [`Halo Visualization`](halo_visualization.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Dynamic Crop`](dynamic_crop.md), [`Image Blur`](image_blur.md), [`LMM`](lmm.md), [`Seg Preview`](seg_preview.md), [`SAM 3`](sam3.md), [`Ellipse Visualization`](ellipse_visualization.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`Line Counter`](line_counter.md), [`LMM For Classification`](lmm_for_classification.md), [`Time in Zone`](timein_zone.md), [`Cache Set`](cache_set.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Event Writer`](event_writer.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`OpenAI`](open_ai.md), [`Path Deviation`](path_deviation.md), [`Time in Zone`](timein_zone.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Google Gemma`](google_gemma.md), [`Slack Notification`](slack_notification.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Email Notification`](email_notification.md), [`Circle Visualization`](circle_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Detections Consensus`](detections_consensus.md), [`Crop Visualization`](crop_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Line Counter`](line_counter.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`SAM 3`](sam3.md), [`Object Detection Model`](object_detection_model.md), [`Google Gemini`](google_gemini.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Size Measurement`](size_measurement.md), [`Local File Sink`](local_file_sink.md), [`Object Detection Model`](object_detection_model.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Dot Visualization`](dot_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Time in Zone`](timein_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Depth Estimation`](depth_estimation.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Blur Visualization`](blur_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Color Visualization`](color_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Motion Detection`](motion_detection.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Trace Visualization`](trace_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Current Time`](current_time.md), [`Cache Get`](cache_get.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Google Gemini`](google_gemini.md), [`Detections Stitch`](detections_stitch.md), [`Image Stack`](image_stack.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen-VL`](qwen_vl.md), [`Mask Visualization`](mask_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Text Display`](text_display.md), [`Gaze Detection`](gaze_detection.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Vision OCR`](google_vision_ocr.md), [`Webhook Sink`](webhook_sink.md), [`OpenAI`](open_ai.md), [`Pixelate Visualization`](pixelate_visualization.md), [`S3 Sink`](s3_sink.md), [`Classification Label Visualization`](classification_label_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Polygon Visualization`](polygon_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Morphological Transformation`](morphological_transformation.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Template Matching`](template_matching.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Camera Calibration`](camera_calibration.md), [`Morphological Transformation`](morphological_transformation.md), [`Path Deviation`](path_deviation.md), [`Background Color Visualization`](background_color_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Clip Comparison`](clip_comparison.md), [`Corner Visualization`](corner_visualization.md), [`Moondream2`](moondream2.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`OpenRouter`](open_router.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Triangle Visualization`](triangle_visualization.md), [`OpenAI`](open_ai.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Pixel Color Count`](pixel_color_count.md), [`Image Preprocessing`](image_preprocessing.md), [`SIFT Comparison`](sift_comparison.md), [`YOLO-World Model`](yolo_world_model.md), [`Halo Visualization`](halo_visualization.md), [`Label Visualization`](label_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`MQTT Writer` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `host` (*[`string`](../kinds/string.md)*): Host of the MQTT broker..
        - `port` (*[`integer`](../kinds/integer.md)*): Port of the MQTT broker..
        - `topic` (*[`string`](../kinds/string.md)*): MQTT topic to publish the message to..
        - `message` (*[`string`](../kinds/string.md)*): Message to be published..
        - `qos` (*[`integer`](../kinds/integer.md)*): Quality of Service level for the message..
        - `retain` (*[`boolean`](../kinds/boolean.md)*): Whether the message should be retained by the broker..
        - `timeout` (*[`float`](../kinds/float.md)*): Timeout for connecting to the MQTT broker and for sending MQTT messages..
        - `username` (*[`string`](../kinds/string.md)*): Username for MQTT broker authentication..
        - `password` (*[`string`](../kinds/string.md)*): Password for MQTT broker authentication..

    - output
    
        - `error_status` ([`boolean`](../kinds/boolean.md)): Boolean flag.
        - `message` ([`string`](../kinds/string.md)): String value.



??? tip "Example JSON definition of step `MQTT Writer` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "mqtt_writer_sink@v1",
	    "host": "localhost",
	    "port": 1883,
	    "topic": "sensors/temperature",
	    "message": "Hello, MQTT!",
	    "qos": 0,
	    "retain": true,
	    "timeout": 0.5,
	    "username": "$inputs.mqtt_username",
	    "password": "$inputs.mqtt_password"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    


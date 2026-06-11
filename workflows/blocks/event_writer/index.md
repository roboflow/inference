
# Event Writer



??? "Class: `EventWriterSinkBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/enterprise/workflows/enterprise_blocks/sinks/event_writer/v1.py">inference.enterprise.workflows.enterprise_blocks.sinks.event_writer.v1.EventWriterSinkBlockV1</a>
    



The **Event Writer** block sends structured events to the Event Ingestion Service
using the v2 API.

## Supported Event Schemas

* **Quality Check** — pass/fail inspection results
* **Inventory Count** — item counts at a location
* **Safety Alert** — safety incidents with severity levels
* **Custom** — free-form events with an arbitrary value field

## Images

Each event includes one image entry. You must provide an **output image** (the
primary display image, typically a visualization). You can optionally attach an
**input image** (the original frame before any annotation).

## Annotations

You can optionally pass object detection, classification, instance segmentation,
or keypoint predictions from upstream model blocks. These are stored as
structured annotations on the image within the event.

## Execution Modes

* **Fire-and-forget** (`fire_and_forget=True`, default) — the HTTP request is
  dispatched in the background so the workflow continues immediately. The
  `event_id` output will be empty.
* **Synchronous** (`fire_and_forget=False`) — the block waits for the response
  and returns the created `event_id`.

## Rate Limiting

Use the **Rate Limiter** workflow block upstream of this block to control how
often events are sent.

## Authentication

If the Event Ingestion Service requires an API key, set the
`EVENT_INGESTION_API_KEY` environment variable on the inference server.
Requests are sent unauthenticated when the variable is not set.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_enterprise/event_writer_sink@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `event_ingestion_url` | `str` | Base URL of the Event Ingestion Service.. | ✅ |
| `event_schema` | `str` | The event schema to use.. | ❌ |
| `image_label` | `str` | Label for the image entry.. | ✅ |
| `custom_metadata` | `Dict[str, Union[bool, float, int, str]]` | Flat key-value metadata (max 100 keys, values must be str/int/float/bool).. | ✅ |
| `qc_result` | `str` | Quality check result: pass or fail.. | ✅ |
| `external_id` | `str` | External identifier for correlation with other systems (max 1000 chars).. | ✅ |
| `location` | `str` | Location identifier for inventory count.. | ✅ |
| `item_count` | `int` | Number of items counted.. | ✅ |
| `item_type` | `str` | Type of item being counted.. | ✅ |
| `alert_type` | `str` | Alert type identifier (alphanumeric, underscores, hyphens).. | ✅ |
| `severity` | `str` | Severity level for the safety alert.. | ✅ |
| `alert_description` | `str` | Description of the safety alert (max 10000 chars).. | ✅ |
| `custom_value` | `str` | Arbitrary value for custom events (max 10000 chars).. | ✅ |
| `fire_and_forget` | `bool` | If True, send the event asynchronously (no event_id returned). If False, wait for the response and return the event_id.. | ✅ |
| `disable_sink` | `bool` | If True, skip sending the event entirely.. | ✅ |
| `request_timeout` | `int` | HTTP request timeout in seconds.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Event Writer` in version `v1`.

    - inputs: [`Camera Focus`](camera_focus.md), [`Current Time`](current_time.md), [`Camera Focus`](camera_focus.md), [`Image Preprocessing`](image_preprocessing.md), [`Background Subtraction`](background_subtraction.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Color Visualization`](color_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Seg Preview`](seg_preview.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`JSON Parser`](json_parser.md), [`Cache Set`](cache_set.md), [`Google Vision OCR`](google_vision_ocr.md), [`SIFT Comparison`](sift_comparison.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Keypoint Visualization`](keypoint_visualization.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Camera Calibration`](camera_calibration.md), [`VLM As Detector`](vlm_as_detector.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Expression`](expression.md), [`LMM For Classification`](lmm_for_classification.md), [`Barcode Detection`](barcode_detection.md), [`Perspective Correction`](perspective_correction.md), [`Cache Get`](cache_get.md), [`Detection Offset`](detection_offset.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Clip Comparison`](clip_comparison.md), [`Qwen-VL`](qwen_vl.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Dot Visualization`](dot_visualization.md), [`OpenAI`](open_ai.md), [`Moondream2`](moondream2.md), [`Size Measurement`](size_measurement.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Dominant Color`](dominant_color.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`SIFT`](sift.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Dimension Collapse`](dimension_collapse.md), [`Email Notification`](email_notification.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Image Slicer`](image_slicer.md), [`Template Matching`](template_matching.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Text Display`](text_display.md), [`Distance Measurement`](distance_measurement.md), [`Time in Zone`](timein_zone.md), [`Blur Visualization`](blur_visualization.md), [`Qwen3-VL`](qwen3_vl.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`LMM`](lmm.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Circle Visualization`](circle_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Label Visualization`](label_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`OpenAI`](open_ai.md), [`CSV Formatter`](csv_formatter.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`CogVLM`](cog_vlm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Detections Merge`](detections_merge.md), [`Clip Comparison`](clip_comparison.md), [`Triangle Visualization`](triangle_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Detections Consensus`](detections_consensus.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Delta Filter`](delta_filter.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`OpenAI`](open_ai.md), [`Email Notification`](email_notification.md), [`Slack Notification`](slack_notification.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detections Filter`](detections_filter.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Google Gemma API`](google_gemma_api.md), [`Event Writer`](event_writer.md), [`Google Gemini`](google_gemini.md), [`Rate Limiter`](rate_limiter.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Inner Workflow`](inner_workflow.md), [`Image Contours`](image_contours.md), [`SAM 3`](sam3.md), [`YOLO-World Model`](yolo_world_model.md), [`Crop Visualization`](crop_visualization.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Relative Static Crop`](relative_static_crop.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`EasyOCR`](easy_ocr.md), [`VLM As Detector`](vlm_as_detector.md), [`SAM 3`](sam3.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Qwen3.5`](qwen3.5.md), [`Google Gemma`](google_gemma.md), [`Google Gemini`](google_gemini.md), [`Image Blur`](image_blur.md), [`Identify Changes`](identify_changes.md), [`MQTT Writer`](mqtt_writer.md), [`Buffer`](buffer.md), [`Anthropic Claude`](anthropic_claude.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`SIFT Comparison`](sift_comparison.md), [`Detections Stitch`](detections_stitch.md), [`Overlap Analysis`](overlap_analysis.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`SAM 3`](sam3.md), [`Time in Zone`](timein_zone.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Contrast Equalization`](contrast_equalization.md), [`SORT Tracker`](sort_tracker.md), [`Image Stack`](image_stack.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Identify Outliers`](identify_outliers.md), [`Dynamic Zone`](dynamic_zone.md), [`Continue If`](continue_if.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`QR Code Detection`](qr_code_detection.md), [`Property Definition`](property_definition.md), [`Path Deviation`](path_deviation.md), [`Webhook Sink`](webhook_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Image Slicer`](image_slicer.md), [`Halo Visualization`](halo_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Local File Sink`](local_file_sink.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`SmolVLM2`](smol_vlm2.md), [`Path Deviation`](path_deviation.md), [`Depth Estimation`](depth_estimation.md), [`Byte Tracker`](byte_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Icon Visualization`](icon_visualization.md), [`Motion Detection`](motion_detection.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Anthropic Claude`](anthropic_claude.md), [`Grid Visualization`](grid_visualization.md), [`Cosine Similarity`](cosine_similarity.md), [`OpenRouter`](open_router.md), [`Anthropic Claude`](anthropic_claude.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Image Threshold`](image_threshold.md), [`Overlap Filter`](overlap_filter.md), [`Data Aggregator`](data_aggregator.md), [`Trace Visualization`](trace_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Morphological Transformation`](morphological_transformation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Detections Transformation`](detections_transformation.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Gaze Detection`](gaze_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Byte Tracker`](byte_tracker.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Stitch Images`](stitch_images.md), [`Florence-2 Model`](florence2_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Byte Tracker`](byte_tracker.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Corner Visualization`](corner_visualization.md), [`Detections Combine`](detections_combine.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`OCR Model`](ocr_model.md), [`Line Counter`](line_counter.md), [`S3 Sink`](s3_sink.md), [`GLM-OCR`](glmocr.md), [`Detection Event Log`](detection_event_log.md), [`OpenAI`](open_ai.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Velocity`](velocity.md)
    - outputs: [`Google Gemma API`](google_gemma_api.md), [`Event Writer`](event_writer.md), [`Google Gemini`](google_gemini.md), [`Current Time`](current_time.md), [`Ellipse Visualization`](ellipse_visualization.md), [`SAM 3`](sam3.md), [`Image Preprocessing`](image_preprocessing.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Color Visualization`](color_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`YOLO-World Model`](yolo_world_model.md), [`Seg Preview`](seg_preview.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Cache Set`](cache_set.md), [`Google Vision OCR`](google_vision_ocr.md), [`SIFT Comparison`](sift_comparison.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Object Detection Model`](object_detection_model.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`SAM 3`](sam3.md), [`Google Gemma`](google_gemma.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Blur`](image_blur.md), [`Keypoint Visualization`](keypoint_visualization.md), [`MQTT Writer`](mqtt_writer.md), [`Anthropic Claude`](anthropic_claude.md), [`Camera Calibration`](camera_calibration.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Detections Stitch`](detections_stitch.md), [`LMM For Classification`](lmm_for_classification.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Perspective Correction`](perspective_correction.md), [`Cache Get`](cache_get.md), [`Time in Zone`](timein_zone.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`SAM 3`](sam3.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Contrast Equalization`](contrast_equalization.md), [`Clip Comparison`](clip_comparison.md), [`Qwen-VL`](qwen_vl.md), [`Image Stack`](image_stack.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Dot Visualization`](dot_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`OpenAI`](open_ai.md), [`Moondream2`](moondream2.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Size Measurement`](size_measurement.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Path Deviation`](path_deviation.md), [`Webhook Sink`](webhook_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Halo Visualization`](halo_visualization.md), [`Email Notification`](email_notification.md), [`QR Code Generator`](qr_code_generator.md), [`Local File Sink`](local_file_sink.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Object Detection Model`](object_detection_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Path Deviation`](path_deviation.md), [`Template Matching`](template_matching.md), [`Depth Estimation`](depth_estimation.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Text Display`](text_display.md), [`Time in Zone`](timein_zone.md), [`Object Detection Model`](object_detection_model.md), [`Distance Measurement`](distance_measurement.md), [`Icon Visualization`](icon_visualization.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`OpenRouter`](open_router.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Anthropic Claude`](anthropic_claude.md), [`LMM`](lmm.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Image Threshold`](image_threshold.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Label Visualization`](label_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Morphological Transformation`](morphological_transformation.md), [`Morphological Transformation`](morphological_transformation.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Gaze Detection`](gaze_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`CogVLM`](cog_vlm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Florence-2 Model`](florence2_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Background Color Visualization`](background_color_visualization.md), [`Detections Consensus`](detections_consensus.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Corner Visualization`](corner_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Line Counter Visualization`](line_counter_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Halo Visualization`](halo_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Line Counter`](line_counter.md), [`OpenAI`](open_ai.md), [`Email Notification`](email_notification.md), [`Slack Notification`](slack_notification.md), [`S3 Sink`](s3_sink.md), [`GLM-OCR`](glmocr.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OPC UA Writer Sink`](opcua_writer_sink.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Event Writer` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `event_ingestion_url` (*[`string`](../kinds/string.md)*): Base URL of the Event Ingestion Service..
        - `output_image` (*[`image`](../kinds/image.md)*): The output/visualization image. Sent as the primary display image..
        - `input_image` (*[`image`](../kinds/image.md)*): The original input image (optional). Sent as the source image..
        - `image_label` (*[`string`](../kinds/string.md)*): Label for the image entry..
        - `custom_metadata` (*[`*`](../kinds/wildcard.md)*): Flat key-value metadata (max 100 keys, values must be str/int/float/bool)..
        - `qc_result` (*[`string`](../kinds/string.md)*): Quality check result: pass or fail..
        - `external_id` (*[`string`](../kinds/string.md)*): External identifier for correlation with other systems (max 1000 chars)..
        - `location` (*[`string`](../kinds/string.md)*): Location identifier for inventory count..
        - `item_count` (*[`integer`](../kinds/integer.md)*): Number of items counted..
        - `item_type` (*[`string`](../kinds/string.md)*): Type of item being counted..
        - `alert_type` (*[`string`](../kinds/string.md)*): Alert type identifier (alphanumeric, underscores, hyphens)..
        - `severity` (*[`string`](../kinds/string.md)*): Severity level for the safety alert..
        - `alert_description` (*[`string`](../kinds/string.md)*): Description of the safety alert (max 10000 chars)..
        - `custom_value` (*[`string`](../kinds/string.md)*): Arbitrary value for custom events (max 10000 chars)..
        - `object_detections` (*[`object_detection_prediction`](../kinds/object_detection_prediction.md)*): Object detection predictions to attach to the image..
        - `classifications` (*[`classification_prediction`](../kinds/classification_prediction.md)*): Classification predictions to attach to the image..
        - `instance_segmentations` (*[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)*): Instance segmentation predictions to attach to the image..
        - `keypoint_detections` (*[`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md)*): Keypoint detection predictions to attach to the image..
        - `fire_and_forget` (*[`boolean`](../kinds/boolean.md)*): If True, send the event asynchronously (no event_id returned). If False, wait for the response and return the event_id..
        - `disable_sink` (*[`boolean`](../kinds/boolean.md)*): If True, skip sending the event entirely..
        - `request_timeout` (*[`integer`](../kinds/integer.md)*): HTTP request timeout in seconds..

    - output
    
        - `error_status` ([`boolean`](../kinds/boolean.md)): Boolean flag.
        - `event_id` ([`string`](../kinds/string.md)): String value.
        - `message` ([`string`](../kinds/string.md)): String value.



??? tip "Example JSON definition of step `Event Writer` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_enterprise/event_writer_sink@v1",
	    "event_ingestion_url": "http://localhost:8001",
	    "event_schema": "<block_does_not_provide_example>",
	    "output_image": "<block_does_not_provide_example>",
	    "input_image": "<block_does_not_provide_example>",
	    "image_label": "defect-analysis",
	    "custom_metadata": {
	        "line": "A1",
	        "shift": "morning"
	    },
	    "qc_result": "pass",
	    "external_id": "batch-2025-001",
	    "location": "warehouse-A",
	    "item_count": 42,
	    "item_type": "widget",
	    "alert_type": "no_hardhat",
	    "severity": "high",
	    "alert_description": "Worker detected without hardhat in zone B",
	    "custom_value": "anomaly detected at 14:32",
	    "object_detections": "<block_does_not_provide_example>",
	    "classifications": "<block_does_not_provide_example>",
	    "instance_segmentations": "<block_does_not_provide_example>",
	    "keypoint_detections": "<block_does_not_provide_example>",
	    "fire_and_forget": true,
	    "disable_sink": false,
	    "request_timeout": 5
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    



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

    - inputs: [`Image Preprocessing`](image_preprocessing.md), [`Detections Transformation`](detections_transformation.md), [`Object Detection Model`](object_detection_model.md), [`Time in Zone`](timein_zone.md), [`Text Display`](text_display.md), [`Template Matching`](template_matching.md), [`Image Threshold`](image_threshold.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Time in Zone`](timein_zone.md), [`OpenAI`](open_ai.md), [`Crop Visualization`](crop_visualization.md), [`Cosine Similarity`](cosine_similarity.md), [`Florence-2 Model`](florence2_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Polygon Visualization`](polygon_visualization.md), [`S3 Sink`](s3_sink.md), [`Absolute Static Crop`](absolute_static_crop.md), [`QR Code Generator`](qr_code_generator.md), [`SIFT Comparison`](sift_comparison.md), [`Cache Get`](cache_get.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OCR Model`](ocr_model.md), [`Color Visualization`](color_visualization.md), [`Gaze Detection`](gaze_detection.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Blur`](image_blur.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Blur Visualization`](blur_visualization.md), [`SAM 3`](sam3.md), [`Detection Offset`](detection_offset.md), [`Anthropic Claude`](anthropic_claude.md), [`MQTT Writer`](mqtt_writer.md), [`Image Slicer`](image_slicer.md), [`SAM 3`](sam3.md), [`Detections Consensus`](detections_consensus.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Detections Stitch`](detections_stitch.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Overlap Analysis`](overlap_analysis.md), [`Rate Limiter`](rate_limiter.md), [`Image Stack`](image_stack.md), [`Google Gemini`](google_gemini.md), [`Delta Filter`](delta_filter.md), [`Cache Set`](cache_set.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Camera Focus`](camera_focus.md), [`CSV Formatter`](csv_formatter.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Seg Preview`](seg_preview.md), [`Overlap Filter`](overlap_filter.md), [`Buffer`](buffer.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Icon Visualization`](icon_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Dominant Color`](dominant_color.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Distance Measurement`](distance_measurement.md), [`Qwen3-VL`](qwen3_vl.md), [`Background Color Visualization`](background_color_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Clip Comparison`](clip_comparison.md), [`Corner Visualization`](corner_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Image Slicer`](image_slicer.md), [`Dynamic Crop`](dynamic_crop.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Detection Event Log`](detection_event_log.md), [`VLM As Detector`](vlm_as_detector.md), [`Anthropic Claude`](anthropic_claude.md), [`Clip Comparison`](clip_comparison.md), [`SORT Tracker`](sort_tracker.md), [`OpenAI`](open_ai.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Detections Combine`](detections_combine.md), [`Camera Calibration`](camera_calibration.md), [`Trace Visualization`](trace_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Email Notification`](email_notification.md), [`Camera Focus`](camera_focus.md), [`Background Subtraction`](background_subtraction.md), [`GLM-OCR`](glmocr.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Qwen3.5`](qwen3.5.md), [`VLM As Detector`](vlm_as_detector.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Property Definition`](property_definition.md), [`Stitch Images`](stitch_images.md), [`Mask Visualization`](mask_visualization.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Data Aggregator`](data_aggregator.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Morphological Transformation`](morphological_transformation.md), [`Email Notification`](email_notification.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Halo Visualization`](halo_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Pixel Color Count`](pixel_color_count.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Qwen-VL`](qwen_vl.md), [`CogVLM`](cog_vlm.md), [`SAM 3`](sam3.md), [`Dot Visualization`](dot_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`Detections Merge`](detections_merge.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Dimension Collapse`](dimension_collapse.md), [`Mask Edge Snap`](mask_edge_snap.md), [`SIFT Comparison`](sift_comparison.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Dynamic Zone`](dynamic_zone.md), [`Detections Filter`](detections_filter.md), [`Byte Tracker`](byte_tracker.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Environment Secrets Store`](environment_secrets_store.md), [`LMM For Classification`](lmm_for_classification.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Object Detection Model`](object_detection_model.md), [`Path Deviation`](path_deviation.md), [`Current Time`](current_time.md), [`Perspective Correction`](perspective_correction.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Byte Tracker`](byte_tracker.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemini`](google_gemini.md), [`Identify Changes`](identify_changes.md), [`Depth Estimation`](depth_estimation.md), [`Object Detection Model`](object_detection_model.md), [`Slack Notification`](slack_notification.md), [`Identify Outliers`](identify_outliers.md), [`Inner Workflow`](inner_workflow.md), [`Time in Zone`](timein_zone.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Label Visualization`](label_visualization.md), [`Size Measurement`](size_measurement.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI`](open_ai.md), [`SIFT`](sift.md), [`Anthropic Claude`](anthropic_claude.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Moondream2`](moondream2.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Florence-2 Model`](florence2_model.md), [`EasyOCR`](easy_ocr.md), [`YOLO-World Model`](yolo_world_model.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Local File Sink`](local_file_sink.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Path Deviation`](path_deviation.md), [`JSON Parser`](json_parser.md), [`OpenRouter`](open_router.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Continue If`](continue_if.md), [`Google Gemini`](google_gemini.md), [`Grid Visualization`](grid_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Reference Path Visualization`](reference_path_visualization.md), [`SmolVLM2`](smol_vlm2.md), [`Line Counter`](line_counter.md), [`Halo Visualization`](halo_visualization.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Webhook Sink`](webhook_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Expression`](expression.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Barcode Detection`](barcode_detection.md), [`Velocity`](velocity.md), [`Motion Detection`](motion_detection.md), [`Google Gemma`](google_gemma.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Line Counter`](line_counter.md), [`QR Code Detection`](qr_code_detection.md), [`Circle Visualization`](circle_visualization.md), [`LMM`](lmm.md), [`Event Writer`](event_writer.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Contrast Equalization`](contrast_equalization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Image Contours`](image_contours.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Polygon Visualization`](polygon_visualization.md)
    - outputs: [`Detections Classes Replacement`](detections_classes_replacement.md), [`Morphological Transformation`](morphological_transformation.md), [`Image Preprocessing`](image_preprocessing.md), [`Email Notification`](email_notification.md), [`Halo Visualization`](halo_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Object Detection Model`](object_detection_model.md), [`Time in Zone`](timein_zone.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Text Display`](text_display.md), [`Template Matching`](template_matching.md), [`Pixel Color Count`](pixel_color_count.md), [`Image Threshold`](image_threshold.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Time in Zone`](timein_zone.md), [`Qwen-VL`](qwen_vl.md), [`OpenAI`](open_ai.md), [`CogVLM`](cog_vlm.md), [`Crop Visualization`](crop_visualization.md), [`SAM 3`](sam3.md), [`Dot Visualization`](dot_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`Florence-2 Model`](florence2_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`S3 Sink`](s3_sink.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`QR Code Generator`](qr_code_generator.md), [`SIFT Comparison`](sift_comparison.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Cache Get`](cache_get.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Dynamic Zone`](dynamic_zone.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Gaze Detection`](gaze_detection.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`LMM For Classification`](lmm_for_classification.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Blur`](image_blur.md), [`Object Detection Model`](object_detection_model.md), [`Blur Visualization`](blur_visualization.md), [`Path Deviation`](path_deviation.md), [`Current Time`](current_time.md), [`SAM 3`](sam3.md), [`Perspective Correction`](perspective_correction.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`MQTT Writer`](mqtt_writer.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemini`](google_gemini.md), [`SAM 3`](sam3.md), [`Depth Estimation`](depth_estimation.md), [`Detections Consensus`](detections_consensus.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Google Gemma API`](google_gemma_api.md), [`Object Detection Model`](object_detection_model.md), [`Slack Notification`](slack_notification.md), [`Time in Zone`](timein_zone.md), [`Image Stack`](image_stack.md), [`Google Gemini`](google_gemini.md), [`Cache Set`](cache_set.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Label Visualization`](label_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Size Measurement`](size_measurement.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI`](open_ai.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Moondream2`](moondream2.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Florence-2 Model`](florence2_model.md), [`Seg Preview`](seg_preview.md), [`YOLO-World Model`](yolo_world_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Local File Sink`](local_file_sink.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Triangle Visualization`](triangle_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Path Deviation`](path_deviation.md), [`OpenRouter`](open_router.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Distance Measurement`](distance_measurement.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OpenAI`](open_ai.md), [`Background Color Visualization`](background_color_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemini`](google_gemini.md), [`Corner Visualization`](corner_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Line Counter`](line_counter.md), [`Halo Visualization`](halo_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Dynamic Crop`](dynamic_crop.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Motion Detection`](motion_detection.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Camera Calibration`](camera_calibration.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Google Gemma`](google_gemma.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Line Counter`](line_counter.md), [`Circle Visualization`](circle_visualization.md), [`Email Notification`](email_notification.md), [`LMM`](lmm.md), [`Event Writer`](event_writer.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`GLM-OCR`](glmocr.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Mask Visualization`](mask_visualization.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Polygon Visualization`](polygon_visualization.md)

    
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



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

    - inputs: [`PLC Writer`](plc_writer.md), [`Image Blur`](image_blur.md), [`Track Class Lock`](track_class_lock.md), [`Byte Tracker`](byte_tracker.md), [`Mask Edge Snap`](mask_edge_snap.md), [`VLM As Detector`](vlm_as_detector.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Image Slicer`](image_slicer.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Motion Detection`](motion_detection.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Clip Comparison`](clip_comparison.md), [`Buffer`](buffer.md), [`Label Visualization`](label_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`S3 Sink`](s3_sink.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Velocity`](velocity.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`CSV Formatter`](csv_formatter.md), [`Background Subtraction`](background_subtraction.md), [`Corner Visualization`](corner_visualization.md), [`Detection Offset`](detection_offset.md), [`JSON Parser`](json_parser.md), [`Property Definition`](property_definition.md), [`Google Gemini`](google_gemini.md), [`Seg Preview`](seg_preview.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`LMM For Classification`](lmm_for_classification.md), [`PP-OCR`](ppocr.md), [`SIFT Comparison`](sift_comparison.md), [`Color Visualization`](color_visualization.md), [`Inner Workflow`](inner_workflow.md), [`Object Detection Model`](object_detection_model.md), [`Local File Sink`](local_file_sink.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Dynamic Zone`](dynamic_zone.md), [`Google Gemma API`](google_gemma_api.md), [`Detections Filter`](detections_filter.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Icon Visualization`](icon_visualization.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Detections Transformation`](detections_transformation.md), [`Rich Label Visualization`](rich_label_visualization.md), [`Identify Changes`](identify_changes.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Expression`](expression.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Google Gemini`](google_gemini.md), [`Detections Stitch`](detections_stitch.md), [`Event Writer`](event_writer.md), [`CogVLM`](cog_vlm.md), [`Time in Zone`](timein_zone.md), [`Email Notification`](email_notification.md), [`Current Time`](current_time.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Perspective Correction`](perspective_correction.md), [`Cosine Similarity`](cosine_similarity.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Overlap Filter`](overlap_filter.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Cache Get`](cache_get.md), [`OpenRouter`](open_router.md), [`Detections Consensus`](detections_consensus.md), [`OCR Model`](ocr_model.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Time in Zone`](timein_zone.md), [`Relative Static Crop`](relative_static_crop.md), [`SAM 3`](sam3.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`GLM-OCR`](glmocr.md), [`Florence-2 Model`](florence2_model.md), [`Background Color Visualization`](background_color_visualization.md), [`Qwen3.5`](qwen3.5.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Threshold`](image_threshold.md), [`Camera Focus`](camera_focus.md), [`Line Counter`](line_counter.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Circle Visualization`](circle_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Detections Merge`](detections_merge.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Cosmos 3`](cosmos3.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`MQTT Writer`](mqtt_writer.md), [`SAM 3`](sam3.md), [`Line Counter`](line_counter.md), [`Image Stack`](image_stack.md), [`Grid Visualization`](grid_visualization.md), [`Florence-2 Model`](florence2_model.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Byte Tracker`](byte_tracker.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Path Deviation`](path_deviation.md), [`Crop Visualization`](crop_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Webhook Sink`](webhook_sink.md), [`VLM As Classifier`](vlm_as_classifier.md), [`SAM 3 Interactive`](sam3_interactive.md), [`YOLO-World Model`](yolo_world_model.md), [`SIFT`](sift.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Slack Notification`](slack_notification.md), [`PLC Reader`](plc_reader.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Label Visualization`](label_visualization.md), [`Email Notification`](email_notification.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Overlap Analysis`](overlap_analysis.md), [`Pixel Color Count`](pixel_color_count.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Stitch Images`](stitch_images.md), [`Path Deviation`](path_deviation.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Distance Measurement`](distance_measurement.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Camera Calibration`](camera_calibration.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Camera Focus`](camera_focus.md), [`Delta Filter`](delta_filter.md), [`Trace Visualization`](trace_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Morphological Transformation`](morphological_transformation.md), [`Dominant Color`](dominant_color.md), [`Continue If`](continue_if.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Google Vision OCR`](google_vision_ocr.md), [`Google Gemma`](google_gemma.md), [`Polygon Visualization`](polygon_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Image Slicer`](image_slicer.md), [`OpenAI`](open_ai.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Object Detection Model`](object_detection_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Mask Visualization`](mask_visualization.md), [`Moondream2`](moondream2.md), [`Nearest Neighbor Detection Match`](nearest_neighbor_detection_match.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Byte Tracker`](byte_tracker.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`VLM As Detector`](vlm_as_detector.md), [`Blur Visualization`](blur_visualization.md), [`Frame Delay`](frame_delay.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Switch Case`](switch_case.md), [`Cache Set`](cache_set.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Barcode Detection`](barcode_detection.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Template Matching`](template_matching.md), [`QR Code Detection`](qr_code_detection.md), [`Halo Visualization`](halo_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`Dynamic Crop`](dynamic_crop.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`GeoTag Detection`](geo_tag_detection.md), [`Halo Visualization`](halo_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Time in Zone`](timein_zone.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Size Measurement`](size_measurement.md), [`LMM`](lmm.md), [`Depth Estimation`](depth_estimation.md), [`SIFT Comparison`](sift_comparison.md), [`Detections Combine`](detections_combine.md), [`Dot Visualization`](dot_visualization.md), [`EasyOCR`](easy_ocr.md), [`Rate Limiter`](rate_limiter.md), [`Google Gemini`](google_gemini.md), [`Data Aggregator`](data_aggregator.md), [`Image Contours`](image_contours.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Morphological Transformation`](morphological_transformation.md), [`SAM 3`](sam3.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Gemini`](google_gemini.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Anthropic Claude`](anthropic_claude.md), [`SORT Tracker`](sort_tracker.md), [`Image Preprocessing`](image_preprocessing.md), [`Qwen3-VL`](qwen3_vl.md), [`Gaze Detection`](gaze_detection.md), [`Anthropic Claude`](anthropic_claude.md), [`Text Display`](text_display.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Identify Outliers`](identify_outliers.md), [`Classification Label Visualization`](classification_label_visualization.md), [`SmolVLM2`](smol_vlm2.md)
    - outputs: [`PLC Writer`](plc_writer.md), [`Image Blur`](image_blur.md), [`Path Deviation`](path_deviation.md), [`Crop Visualization`](crop_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Webhook Sink`](webhook_sink.md), [`Motion Detection`](motion_detection.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`YOLO-World Model`](yolo_world_model.md), [`Slack Notification`](slack_notification.md), [`Label Visualization`](label_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Label Visualization`](label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`S3 Sink`](s3_sink.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Email Notification`](email_notification.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Pixel Color Count`](pixel_color_count.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Path Deviation`](path_deviation.md), [`Corner Visualization`](corner_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Distance Measurement`](distance_measurement.md), [`Camera Calibration`](camera_calibration.md), [`Google Gemini`](google_gemini.md), [`Seg Preview`](seg_preview.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`LMM For Classification`](lmm_for_classification.md), [`SIFT Comparison`](sift_comparison.md), [`Trace Visualization`](trace_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Color Visualization`](color_visualization.md), [`Local File Sink`](local_file_sink.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Dynamic Zone`](dynamic_zone.md), [`Morphological Transformation`](morphological_transformation.md), [`Google Vision OCR`](google_vision_ocr.md), [`Google Gemma API`](google_gemma_api.md), [`Google Gemma`](google_gemma.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Polygon Visualization`](polygon_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Rich Label Visualization`](rich_label_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Keypoint Visualization`](keypoint_visualization.md), [`OpenAI`](open_ai.md), [`Object Detection Model`](object_detection_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Mask Visualization`](mask_visualization.md), [`Moondream2`](moondream2.md), [`Nearest Neighbor Detection Match`](nearest_neighbor_detection_match.md), [`OpenAI`](open_ai.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Blur Visualization`](blur_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Reference Path Visualization`](reference_path_visualization.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Gemini`](google_gemini.md), [`Detections Stitch`](detections_stitch.md), [`Event Writer`](event_writer.md), [`Cache Set`](cache_set.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`CogVLM`](cog_vlm.md), [`Time in Zone`](timein_zone.md), [`Email Notification`](email_notification.md), [`Current Time`](current_time.md), [`Template Matching`](template_matching.md), [`Perspective Correction`](perspective_correction.md), [`Halo Visualization`](halo_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Cache Get`](cache_get.md), [`OpenRouter`](open_router.md), [`Dynamic Crop`](dynamic_crop.md), [`Detections Consensus`](detections_consensus.md), [`Halo Visualization`](halo_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Time in Zone`](timein_zone.md), [`Time in Zone`](timein_zone.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Size Measurement`](size_measurement.md), [`LMM`](lmm.md), [`Depth Estimation`](depth_estimation.md), [`SAM 3`](sam3.md), [`Dot Visualization`](dot_visualization.md), [`GLM-OCR`](glmocr.md), [`Florence-2 Model`](florence2_model.md), [`Background Color Visualization`](background_color_visualization.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Contrast Equalization`](contrast_equalization.md), [`Morphological Transformation`](morphological_transformation.md), [`Image Threshold`](image_threshold.md), [`SAM 3`](sam3.md), [`Line Counter`](line_counter.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Circle Visualization`](circle_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Google Gemini`](google_gemini.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Cosmos 3`](cosmos3.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`MQTT Writer`](mqtt_writer.md), [`SAM 3`](sam3.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Preprocessing`](image_preprocessing.md), [`Gaze Detection`](gaze_detection.md), [`Line Counter`](line_counter.md), [`Anthropic Claude`](anthropic_claude.md), [`Text Display`](text_display.md), [`Image Stack`](image_stack.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md)

    
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


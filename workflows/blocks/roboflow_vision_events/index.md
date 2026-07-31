
# Roboflow Vision Events



??? "Class: `RoboflowVisionEventsBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/sinks/roboflow/vision_events/v1.py">inference.core.workflows.core_steps.sinks.roboflow.vision_events.v1.RoboflowVisionEventsBlockV1</a>
    



Send images, model predictions, and event metadata to the Roboflow Vision Events API for
monitoring, quality control, safety alerting, and custom event tracking.

## How This Block Works

This block uploads workflow images and model predictions to the Roboflow Vision Events API,
creating structured events that can be queried, filtered, and visualized in the Roboflow
dashboard.

1. Optionally uploads an input image and/or output image (visualization) to the Vision Events
   image storage via the public API
2. Converts model predictions (object detection, classification, instance segmentation, or
   keypoint detection) into the Vision Events annotation format and attaches them to the
   input image
3. Creates a vision event with the specified event type, use case, event data,
   and custom metadata
4. Enforces a built-in rate limit (`cooldown_seconds`, default 1 second) so
   high-frequency video workflows do not flood the API with an event per frame
5. Supports fire-and-forget mode for non-blocking execution

## Rate Limiting

Video workflows can run many times per second, which by default would send an event
(and its images) for every frame. To prevent this, the block enforces a cooldown
between consecutive events: at most one event per second is sent by default. Events
triggered during the cooldown period are dropped and the `throttling_status` output
is set to `True`.

Adjust `cooldown_seconds` to your needs, or set it to `0` to disable rate limiting
entirely (e.g. for intentionally bursty use cases). The cooldown timer lives in the
block instance, so it throttles long-lived executions such as video processing with
InferencePipeline. Workflows served over HTTP (e.g. `/workflows/run`) create fresh
block instances per request, so the cooldown does not throttle across separate HTTP
calls.

## Deployment Modes

By default this block sends events to the **Roboflow Vision Events API** (cloud /
Serverless API), uploading images and posting the event over the public API.

For edge deployments, enable **Write to Local Event Store** to send events to a
local Event Ingestion Service instead. In this mode images are embedded directly
in the request (no upload step) and the event is posted to `<event store URL>/v2/events`.
The event store URL defaults to `http://localhost:8001` and can be overridden. No
Roboflow API key is required in this mode; if the local service requires
authentication, set the `EVENT_INGESTION_API_KEY` environment variable on the
inference server.

## Event Types

- **quality_check**: Manufacturing/inspection QA with pass/fail result and optional confidence
- **inventory_count**: Inventory tracking with location, item count, and item type
- **safety_alert**: Safety violations with alert type, severity (low/medium/high), and description
- **custom**: User-defined events with a free-form value string
- **operator_feedback**: Operator review/correction of previous events (correct/incorrect/inconclusive)

## Requirements

The default (cloud) mode requires a valid Roboflow API key with `vision-events:write`
scope, configured in your environment or workflow configuration. No Roboflow API key is
needed when **Write to Local Event Store** is enabled (see Deployment Modes above).

## Common Use Cases

- **Quality Control**: Automatically log inspection results with images and detection overlays
- **Safety Monitoring**: Send safety alerts when violations are detected in video streams
- **Production Analytics**: Track inventory counts and production metrics with visual evidence
- **Active Monitoring**: Fire-and-forget event logging from real-time video processing workflows


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/roboflow_vision_events@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `event_type` | `str` | The type of vision event to create.. | ✅ |
| `solution` | `str` | The use case to associate the event with. Events are namespaced by use case within a workspace.. | ✅ |
| `external_id` | `str` | External identifier for correlation with other systems (max 1000 chars).. | ✅ |
| `qc_result` | `str` | Quality check result: pass or fail.. | ✅ |
| `location` | `str` | Location identifier for inventory count.. | ✅ |
| `item_count` | `int` | Number of items counted.. | ✅ |
| `item_type` | `str` | Type of item being counted.. | ✅ |
| `alert_type` | `str` | Alert type identifier (e.g. no_hardhat, spill_detected).. | ✅ |
| `severity` | `str` | Severity level for the safety alert.. | ✅ |
| `alert_description` | `str` | Description of the safety alert.. | ✅ |
| `custom_value` | `str` | Arbitrary value for custom events.. | ✅ |
| `related_event_id` | `str` | The event ID of the event being reviewed.. | ✅ |
| `feedback` | `str` | Operator feedback on the related event.. | ✅ |
| `custom_metadata` | `Dict[str, Union[bool, float, int, str]]` | Flat key-value metadata to attach to the event. Keys must match pattern [a-zA-Z0-9_ -]+ (max 100 chars). String values max 1000 chars.. | ✅ |
| `fire_and_forget` | `bool` | If True, the event is sent asynchronously and the workflow continues without waiting. If False, the block waits for the API response.. | ✅ |
| `disable_sink` | `bool` | If True, the block is disabled and no events are sent.. | ✅ |
| `cooldown_seconds` | `Union[float, int]` | Minimum number of seconds between consecutive events sent by this block. Events triggered during the cooldown period are dropped and the `throttling_status` output is set to True. Defaults to 1 second (at most 1 event per second) so high-frequency video workflows do not flood the Vision Events API with an event per frame. Set to 0 to disable rate limiting for intentionally bursty use cases.. | ✅ |
| `write_to_event_store` | `bool` | If True, send the event to a local Event Ingestion Service (edge deployment) instead of the Roboflow Vision Events API (cloud). Images are embedded in the request and the event is posted to `<Event Store URL>/v2/events`. No Roboflow API key is required in this mode.. | ✅ |
| `event_store_url` | `str` | Base URL of the local Event Ingestion Service. Only used when `Write to Local Event Store` is enabled.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-alert-circle-outline:{ style="color: #f57c00" } `soft` — runtime `hosted_serverless`, `dedicated_deployment`; execution `remote`
:   Cooldown / rate-limit timer is stored in process memory. With remote step execution on stateless or multi-replica HTTP runtimes each request gets a fresh worker, so cooldown does not throttle. Cooldown only behaves as documented with local step execution inside an InferencePipeline.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Roboflow Vision Events` in version `v1`.

    - inputs: [`PLC Writer`](plc_writer.md), [`Image Blur`](image_blur.md), [`Track Class Lock`](track_class_lock.md), [`Byte Tracker`](byte_tracker.md), [`Mask Edge Snap`](mask_edge_snap.md), [`VLM As Detector`](vlm_as_detector.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Image Slicer`](image_slicer.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Motion Detection`](motion_detection.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Clip Comparison`](clip_comparison.md), [`Buffer`](buffer.md), [`Label Visualization`](label_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`S3 Sink`](s3_sink.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Velocity`](velocity.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Background Subtraction`](background_subtraction.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`CSV Formatter`](csv_formatter.md), [`Corner Visualization`](corner_visualization.md), [`Detection Offset`](detection_offset.md), [`JSON Parser`](json_parser.md), [`Property Definition`](property_definition.md), [`Seg Preview`](seg_preview.md), [`Google Gemini`](google_gemini.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`LMM For Classification`](lmm_for_classification.md), [`PP-OCR`](ppocr.md), [`SIFT Comparison`](sift_comparison.md), [`Color Visualization`](color_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Inner Workflow`](inner_workflow.md), [`Local File Sink`](local_file_sink.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Dynamic Zone`](dynamic_zone.md), [`Google Gemma API`](google_gemma_api.md), [`Detections Filter`](detections_filter.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Icon Visualization`](icon_visualization.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Detections Transformation`](detections_transformation.md), [`Rich Label Visualization`](rich_label_visualization.md), [`Identify Changes`](identify_changes.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Expression`](expression.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Google Gemini`](google_gemini.md), [`Detections Stitch`](detections_stitch.md), [`Event Writer`](event_writer.md), [`CogVLM`](cog_vlm.md), [`Time in Zone`](timein_zone.md), [`Email Notification`](email_notification.md), [`Current Time`](current_time.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Perspective Correction`](perspective_correction.md), [`Cosine Similarity`](cosine_similarity.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Overlap Filter`](overlap_filter.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Cache Get`](cache_get.md), [`OpenRouter`](open_router.md), [`Detections Consensus`](detections_consensus.md), [`OCR Model`](ocr_model.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Time in Zone`](timein_zone.md), [`Relative Static Crop`](relative_static_crop.md), [`SAM 3`](sam3.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`GLM-OCR`](glmocr.md), [`Florence-2 Model`](florence2_model.md), [`Background Color Visualization`](background_color_visualization.md), [`Qwen3.5`](qwen3.5.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Threshold`](image_threshold.md), [`Camera Focus`](camera_focus.md), [`Line Counter`](line_counter.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Circle Visualization`](circle_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Detections Merge`](detections_merge.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`Cosmos 3`](cosmos3.md), [`MQTT Writer`](mqtt_writer.md), [`SAM 3`](sam3.md), [`Line Counter`](line_counter.md), [`Image Stack`](image_stack.md), [`Grid Visualization`](grid_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Florence-2 Model`](florence2_model.md), [`Byte Tracker`](byte_tracker.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Path Deviation`](path_deviation.md), [`Crop Visualization`](crop_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Webhook Sink`](webhook_sink.md), [`VLM As Classifier`](vlm_as_classifier.md), [`SAM 3 Interactive`](sam3_interactive.md), [`YOLO-World Model`](yolo_world_model.md), [`SIFT`](sift.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Slack Notification`](slack_notification.md), [`PLC Reader`](plc_reader.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Label Visualization`](label_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Email Notification`](email_notification.md), [`Overlap Analysis`](overlap_analysis.md), [`Pixel Color Count`](pixel_color_count.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Stitch Images`](stitch_images.md), [`Path Deviation`](path_deviation.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Distance Measurement`](distance_measurement.md), [`Camera Calibration`](camera_calibration.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Camera Focus`](camera_focus.md), [`Delta Filter`](delta_filter.md), [`Trace Visualization`](trace_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Morphological Transformation`](morphological_transformation.md), [`Dominant Color`](dominant_color.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Continue If`](continue_if.md), [`Google Vision OCR`](google_vision_ocr.md), [`Google Gemma`](google_gemma.md), [`Polygon Visualization`](polygon_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Image Slicer`](image_slicer.md), [`Object Detection Model`](object_detection_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`OpenAI`](open_ai.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Mask Visualization`](mask_visualization.md), [`Moondream2`](moondream2.md), [`Nearest Neighbor Detection Match`](nearest_neighbor_detection_match.md), [`Byte Tracker`](byte_tracker.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`VLM As Detector`](vlm_as_detector.md), [`Blur Visualization`](blur_visualization.md), [`Frame Delay`](frame_delay.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Switch Case`](switch_case.md), [`Cache Set`](cache_set.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Barcode Detection`](barcode_detection.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Template Matching`](template_matching.md), [`QR Code Detection`](qr_code_detection.md), [`Halo Visualization`](halo_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`Dynamic Crop`](dynamic_crop.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`GeoTag Detection`](geo_tag_detection.md), [`Halo Visualization`](halo_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Time in Zone`](timein_zone.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Size Measurement`](size_measurement.md), [`LMM`](lmm.md), [`Depth Estimation`](depth_estimation.md), [`Detections Combine`](detections_combine.md), [`SIFT Comparison`](sift_comparison.md), [`Dot Visualization`](dot_visualization.md), [`EasyOCR`](easy_ocr.md), [`Rate Limiter`](rate_limiter.md), [`Google Gemini`](google_gemini.md), [`Data Aggregator`](data_aggregator.md), [`Image Contours`](image_contours.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Morphological Transformation`](morphological_transformation.md), [`SAM 3`](sam3.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Gemini`](google_gemini.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Anthropic Claude`](anthropic_claude.md), [`SORT Tracker`](sort_tracker.md), [`Image Preprocessing`](image_preprocessing.md), [`Qwen3-VL`](qwen3_vl.md), [`Gaze Detection`](gaze_detection.md), [`Anthropic Claude`](anthropic_claude.md), [`Text Display`](text_display.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Identify Outliers`](identify_outliers.md), [`Classification Label Visualization`](classification_label_visualization.md), [`SmolVLM2`](smol_vlm2.md)
    - outputs: [`PLC Writer`](plc_writer.md), [`Image Blur`](image_blur.md), [`Path Deviation`](path_deviation.md), [`Crop Visualization`](crop_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Webhook Sink`](webhook_sink.md), [`Motion Detection`](motion_detection.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`YOLO-World Model`](yolo_world_model.md), [`Slack Notification`](slack_notification.md), [`Label Visualization`](label_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Label Visualization`](label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`S3 Sink`](s3_sink.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Email Notification`](email_notification.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Pixel Color Count`](pixel_color_count.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Path Deviation`](path_deviation.md), [`Corner Visualization`](corner_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Distance Measurement`](distance_measurement.md), [`Camera Calibration`](camera_calibration.md), [`Google Gemini`](google_gemini.md), [`Seg Preview`](seg_preview.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`LMM For Classification`](lmm_for_classification.md), [`SIFT Comparison`](sift_comparison.md), [`Trace Visualization`](trace_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Color Visualization`](color_visualization.md), [`Local File Sink`](local_file_sink.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Dynamic Zone`](dynamic_zone.md), [`Morphological Transformation`](morphological_transformation.md), [`Google Vision OCR`](google_vision_ocr.md), [`Google Gemma API`](google_gemma_api.md), [`Google Gemma`](google_gemma.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Polygon Visualization`](polygon_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Rich Label Visualization`](rich_label_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Keypoint Visualization`](keypoint_visualization.md), [`OpenAI`](open_ai.md), [`Object Detection Model`](object_detection_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Mask Visualization`](mask_visualization.md), [`Moondream2`](moondream2.md), [`Nearest Neighbor Detection Match`](nearest_neighbor_detection_match.md), [`OpenAI`](open_ai.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Blur Visualization`](blur_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Reference Path Visualization`](reference_path_visualization.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Gemini`](google_gemini.md), [`Detections Stitch`](detections_stitch.md), [`Event Writer`](event_writer.md), [`Cache Set`](cache_set.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`CogVLM`](cog_vlm.md), [`Time in Zone`](timein_zone.md), [`Email Notification`](email_notification.md), [`Current Time`](current_time.md), [`Template Matching`](template_matching.md), [`Perspective Correction`](perspective_correction.md), [`Halo Visualization`](halo_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Cache Get`](cache_get.md), [`OpenRouter`](open_router.md), [`Dynamic Crop`](dynamic_crop.md), [`Detections Consensus`](detections_consensus.md), [`Halo Visualization`](halo_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Time in Zone`](timein_zone.md), [`Time in Zone`](timein_zone.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Size Measurement`](size_measurement.md), [`LMM`](lmm.md), [`Depth Estimation`](depth_estimation.md), [`SAM 3`](sam3.md), [`Dot Visualization`](dot_visualization.md), [`GLM-OCR`](glmocr.md), [`Florence-2 Model`](florence2_model.md), [`Background Color Visualization`](background_color_visualization.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Contrast Equalization`](contrast_equalization.md), [`Morphological Transformation`](morphological_transformation.md), [`Image Threshold`](image_threshold.md), [`SAM 3`](sam3.md), [`Line Counter`](line_counter.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Circle Visualization`](circle_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Google Gemini`](google_gemini.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Cosmos 3`](cosmos3.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`MQTT Writer`](mqtt_writer.md), [`SAM 3`](sam3.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Preprocessing`](image_preprocessing.md), [`Gaze Detection`](gaze_detection.md), [`Line Counter`](line_counter.md), [`Anthropic Claude`](anthropic_claude.md), [`Text Display`](text_display.md), [`Image Stack`](image_stack.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Roboflow Vision Events` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `input_image` (*[`image`](../kinds/image.md)*): The original input image. Uploaded to the Vision Events API and used as the base image for detection annotations..
        - `output_image` (*[`image`](../kinds/image.md)*): An optional output/visualized image (e.g., from a visualization block). Displayed as the primary image in the Vision Events dashboard..
        - `predictions` (*Union[[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`classification_prediction`](../kinds/classification_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md)]*): Optional model predictions to include as detection annotations on the input image. Supports object detection, instance segmentation, keypoint detection, and classification predictions..
        - `event_type` (*[`string`](../kinds/string.md)*): The type of vision event to create..
        - `solution` (*Union[[`roboflow_solution`](../kinds/roboflow_solution.md), [`string`](../kinds/string.md)]*): The use case to associate the event with. Events are namespaced by use case within a workspace..
        - `external_id` (*[`string`](../kinds/string.md)*): External identifier for correlation with other systems (max 1000 chars)..
        - `qc_result` (*[`string`](../kinds/string.md)*): Quality check result: pass or fail..
        - `location` (*[`string`](../kinds/string.md)*): Location identifier for inventory count..
        - `item_count` (*[`integer`](../kinds/integer.md)*): Number of items counted..
        - `item_type` (*[`string`](../kinds/string.md)*): Type of item being counted..
        - `alert_type` (*[`string`](../kinds/string.md)*): Alert type identifier (e.g. no_hardhat, spill_detected)..
        - `severity` (*[`string`](../kinds/string.md)*): Severity level for the safety alert..
        - `alert_description` (*[`string`](../kinds/string.md)*): Description of the safety alert..
        - `custom_value` (*[`string`](../kinds/string.md)*): Arbitrary value for custom events..
        - `related_event_id` (*[`string`](../kinds/string.md)*): The event ID of the event being reviewed..
        - `feedback` (*[`string`](../kinds/string.md)*): Operator feedback on the related event..
        - `custom_metadata` (*[`*`](../kinds/wildcard.md)*): Flat key-value metadata to attach to the event. Keys must match pattern [a-zA-Z0-9_ -]+ (max 100 chars). String values max 1000 chars..
        - `fire_and_forget` (*[`boolean`](../kinds/boolean.md)*): If True, the event is sent asynchronously and the workflow continues without waiting. If False, the block waits for the API response..
        - `disable_sink` (*[`boolean`](../kinds/boolean.md)*): If True, the block is disabled and no events are sent..
        - `cooldown_seconds` (*Union[[`integer`](../kinds/integer.md), [`float`](../kinds/float.md)]*): Minimum number of seconds between consecutive events sent by this block. Events triggered during the cooldown period are dropped and the `throttling_status` output is set to True. Defaults to 1 second (at most 1 event per second) so high-frequency video workflows do not flood the Vision Events API with an event per frame. Set to 0 to disable rate limiting for intentionally bursty use cases..
        - `write_to_event_store` (*[`boolean`](../kinds/boolean.md)*): If True, send the event to a local Event Ingestion Service (edge deployment) instead of the Roboflow Vision Events API (cloud). Images are embedded in the request and the event is posted to `<Event Store URL>/v2/events`. No Roboflow API key is required in this mode..
        - `event_store_url` (*[`string`](../kinds/string.md)*): Base URL of the local Event Ingestion Service. Only used when `Write to Local Event Store` is enabled..

    - output
    
        - `error_status` ([`boolean`](../kinds/boolean.md)): Boolean flag.
        - `throttling_status` ([`boolean`](../kinds/boolean.md)): Boolean flag.
        - `event_id` ([`string`](../kinds/string.md)): String value.
        - `message` ([`string`](../kinds/string.md)): String value.



??? tip "Example JSON definition of step `Roboflow Vision Events` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/roboflow_vision_events@v1",
	    "input_image": "$inputs.image",
	    "output_image": "$steps.visualization.image",
	    "predictions": "$steps.object_detection_model.predictions",
	    "event_type": "quality_check",
	    "solution": "my-use-case",
	    "external_id": "batch-2025-001",
	    "qc_result": "pass",
	    "location": "warehouse-A",
	    "item_count": 42,
	    "item_type": "widget",
	    "alert_type": "no_hardhat",
	    "severity": "high",
	    "alert_description": "Worker detected without hardhat in zone B",
	    "custom_value": "anomaly detected at 14:32",
	    "related_event_id": "evt_abc123",
	    "feedback": "correct",
	    "custom_metadata": {
	        "camera_id": "cam_01",
	        "location": "$inputs.location"
	    },
	    "fire_and_forget": true,
	    "disable_sink": false,
	    "cooldown_seconds": 1,
	    "write_to_event_store": false,
	    "event_store_url": "http://localhost:8001"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    


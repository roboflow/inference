
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
4. Supports fire-and-forget mode for non-blocking execution

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
| `write_to_event_store` | `bool` | If True, send the event to a local Event Ingestion Service (edge deployment) instead of the Roboflow Vision Events API (cloud). Images are embedded in the request and the event is posted to `<Event Store URL>/v2/events`. No Roboflow API key is required in this mode.. | ✅ |
| `event_store_url` | `str` | Base URL of the local Event Ingestion Service. Only used when `Write to Local Event Store` is enabled.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Roboflow Vision Events` in version `v1`.

    - inputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Qwen-VL`](qwen_vl.md), [`Slack Notification`](slack_notification.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Mask Edge Snap`](mask_edge_snap.md), [`VLM As Detector`](vlm_as_detector.md), [`Template Matching`](template_matching.md), [`Track Class Lock`](track_class_lock.md), [`Dynamic Zone`](dynamic_zone.md), [`Webhook Sink`](webhook_sink.md), [`SAM 3`](sam3.md), [`Contrast Enhancement`](contrast_enhancement.md), [`QR Code Generator`](qr_code_generator.md), [`SIFT`](sift.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Path Deviation`](path_deviation.md), [`Florence-2 Model`](florence2_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Delta Filter`](delta_filter.md), [`Text Display`](text_display.md), [`MQTT Writer`](mqtt_writer.md), [`Gaze Detection`](gaze_detection.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Expression`](expression.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Time in Zone`](timein_zone.md), [`Time in Zone`](timein_zone.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Image Slicer`](image_slicer.md), [`Background Subtraction`](background_subtraction.md), [`Image Preprocessing`](image_preprocessing.md), [`Byte Tracker`](byte_tracker.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Email Notification`](email_notification.md), [`Line Counter`](line_counter.md), [`LMM`](lmm.md), [`Distance Measurement`](distance_measurement.md), [`Detections Consensus`](detections_consensus.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`SAM 3`](sam3.md), [`Time in Zone`](timein_zone.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Event Writer`](event_writer.md), [`Image Blur`](image_blur.md), [`Google Vision OCR`](google_vision_ocr.md), [`Byte Tracker`](byte_tracker.md), [`Qwen3-VL`](qwen3_vl.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`CSV Formatter`](csv_formatter.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Detections Merge`](detections_merge.md), [`OpenAI`](open_ai.md), [`Mask Visualization`](mask_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Camera Focus`](camera_focus.md), [`Detections Transformation`](detections_transformation.md), [`OpenAI`](open_ai.md), [`LMM For Classification`](lmm_for_classification.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Color Visualization`](color_visualization.md), [`Property Definition`](property_definition.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Object Detection Model`](object_detection_model.md), [`Seg Preview`](seg_preview.md), [`Cache Get`](cache_get.md), [`Camera Calibration`](camera_calibration.md), [`Dominant Color`](dominant_color.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Detections Filter`](detections_filter.md), [`Background Color Visualization`](background_color_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`CogVLM`](cog_vlm.md), [`SORT Tracker`](sort_tracker.md), [`OpenAI`](open_ai.md), [`Identify Outliers`](identify_outliers.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Depth Estimation`](depth_estimation.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Stack`](image_stack.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Google Gemma`](google_gemma.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Detections Stitch`](detections_stitch.md), [`Buffer`](buffer.md), [`Crop Visualization`](crop_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Camera Focus`](camera_focus.md), [`JSON Parser`](json_parser.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`GLM-OCR`](glmocr.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Dimension Collapse`](dimension_collapse.md), [`Continue If`](continue_if.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`VLM As Detector`](vlm_as_detector.md), [`Image Threshold`](image_threshold.md), [`Anthropic Claude`](anthropic_claude.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen3.5`](qwen3.5.md), [`OpenAI`](open_ai.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Detections Combine`](detections_combine.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Cosine Similarity`](cosine_similarity.md), [`Florence-2 Model`](florence2_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Halo Visualization`](halo_visualization.md), [`SAM 3`](sam3.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Halo Visualization`](halo_visualization.md), [`Label Visualization`](label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Byte Tracker`](byte_tracker.md), [`Inner Workflow`](inner_workflow.md), [`Clip Comparison`](clip_comparison.md), [`Detection Offset`](detection_offset.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Morphological Transformation`](morphological_transformation.md), [`PLC Writer`](plc_writer.md), [`Google Gemini`](google_gemini.md), [`PLC Reader`](plc_reader.md), [`Polygon Visualization`](polygon_visualization.md), [`Moondream2`](moondream2.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Google Gemini`](google_gemini.md), [`EasyOCR`](easy_ocr.md), [`Overlap Analysis`](overlap_analysis.md), [`Motion Detection`](motion_detection.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Switch Case`](switch_case.md), [`Data Aggregator`](data_aggregator.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Local File Sink`](local_file_sink.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Rate Limiter`](rate_limiter.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Slicer`](image_slicer.md), [`Identify Changes`](identify_changes.md), [`S3 Sink`](s3_sink.md), [`Cache Set`](cache_set.md), [`Size Measurement`](size_measurement.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Path Deviation`](path_deviation.md), [`Contrast Equalization`](contrast_equalization.md), [`Morphological Transformation`](morphological_transformation.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Barcode Detection`](barcode_detection.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SmolVLM2`](smol_vlm2.md), [`Email Notification`](email_notification.md), [`OpenRouter`](open_router.md), [`Grid Visualization`](grid_visualization.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Velocity`](velocity.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Clip Comparison`](clip_comparison.md), [`Overlap Filter`](overlap_filter.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Corner Visualization`](corner_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Line Counter`](line_counter.md), [`Current Time`](current_time.md), [`Stitch Images`](stitch_images.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Icon Visualization`](icon_visualization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`OCR Model`](ocr_model.md), [`Perspective Correction`](perspective_correction.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Pixel Color Count`](pixel_color_count.md), [`QR Code Detection`](qr_code_detection.md), [`Detection Event Log`](detection_event_log.md), [`Image Contours`](image_contours.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md)
    - outputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Threshold`](image_threshold.md), [`Triangle Visualization`](triangle_visualization.md), [`OpenAI`](open_ai.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Qwen-VL`](qwen_vl.md), [`Slack Notification`](slack_notification.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Florence-2 Model`](florence2_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Template Matching`](template_matching.md), [`Dynamic Zone`](dynamic_zone.md), [`Halo Visualization`](halo_visualization.md), [`Webhook Sink`](webhook_sink.md), [`SAM 3`](sam3.md), [`SAM 3`](sam3.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`QR Code Generator`](qr_code_generator.md), [`Google Gemini`](google_gemini.md), [`YOLO-World Model`](yolo_world_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Halo Visualization`](halo_visualization.md), [`Label Visualization`](label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Path Deviation`](path_deviation.md), [`Florence-2 Model`](florence2_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Morphological Transformation`](morphological_transformation.md), [`Text Display`](text_display.md), [`MQTT Writer`](mqtt_writer.md), [`PLC Writer`](plc_writer.md), [`Gaze Detection`](gaze_detection.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Google Gemini`](google_gemini.md), [`Google Gemma API`](google_gemma_api.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Polygon Visualization`](polygon_visualization.md), [`Moondream2`](moondream2.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Google Gemini`](google_gemini.md), [`Motion Detection`](motion_detection.md), [`Time in Zone`](timein_zone.md), [`Time in Zone`](timein_zone.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Polygon Visualization`](polygon_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Preprocessing`](image_preprocessing.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Local File Sink`](local_file_sink.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Email Notification`](email_notification.md), [`Line Counter`](line_counter.md), [`S3 Sink`](s3_sink.md), [`LMM`](lmm.md), [`Distance Measurement`](distance_measurement.md), [`Detections Consensus`](detections_consensus.md), [`Cache Set`](cache_set.md), [`Size Measurement`](size_measurement.md), [`Trace Visualization`](trace_visualization.md), [`SAM 3`](sam3.md), [`Circle Visualization`](circle_visualization.md), [`Time in Zone`](timein_zone.md), [`Event Writer`](event_writer.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Path Deviation`](path_deviation.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Blur`](image_blur.md), [`Google Vision OCR`](google_vision_ocr.md), [`Morphological Transformation`](morphological_transformation.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Email Notification`](email_notification.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`OpenRouter`](open_router.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`OpenAI`](open_ai.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenAI`](open_ai.md), [`LMM For Classification`](lmm_for_classification.md), [`Anthropic Claude`](anthropic_claude.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Object Detection Model`](object_detection_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Cache Get`](cache_get.md), [`Seg Preview`](seg_preview.md), [`Clip Comparison`](clip_comparison.md), [`Camera Calibration`](camera_calibration.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`CogVLM`](cog_vlm.md), [`Corner Visualization`](corner_visualization.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`OpenAI`](open_ai.md), [`Object Detection Model`](object_detection_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Depth Estimation`](depth_estimation.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Image Stack`](image_stack.md), [`Line Counter`](line_counter.md), [`Current Time`](current_time.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Google Gemma`](google_gemma.md), [`Line Counter Visualization`](line_counter_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Keypoint Visualization`](keypoint_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Detections Stitch`](detections_stitch.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Perspective Correction`](perspective_correction.md), [`Crop Visualization`](crop_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`GLM-OCR`](glmocr.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Pixel Color Count`](pixel_color_count.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Roboflow Vision Events` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `input_image` (*[`image`](../kinds/image.md)*): The original input image. Uploaded to the Vision Events API and used as the base image for detection annotations..
        - `output_image` (*[`image`](../kinds/image.md)*): An optional output/visualized image (e.g., from a visualization block). Displayed as the primary image in the Vision Events dashboard..
        - `predictions` (*Union[[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md), [`classification_prediction`](../kinds/classification_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md)]*): Optional model predictions to include as detection annotations on the input image. Supports object detection, instance segmentation, keypoint detection, and classification predictions..
        - `event_type` (*[`string`](../kinds/string.md)*): The type of vision event to create..
        - `solution` (*Union[[`string`](../kinds/string.md), [`roboflow_solution`](../kinds/roboflow_solution.md)]*): The use case to associate the event with. Events are namespaced by use case within a workspace..
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
        - `write_to_event_store` (*[`boolean`](../kinds/boolean.md)*): If True, send the event to a local Event Ingestion Service (edge deployment) instead of the Roboflow Vision Events API (cloud). Images are embedded in the request and the event is posted to `<Event Store URL>/v2/events`. No Roboflow API key is required in this mode..
        - `event_store_url` (*[`string`](../kinds/string.md)*): Base URL of the local Event Ingestion Service. Only used when `Write to Local Event Store` is enabled..

    - output
    
        - `error_status` ([`boolean`](../kinds/boolean.md)): Boolean flag.
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



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

    - inputs: [`Overlap Analysis`](overlap_analysis.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Detections Transformation`](detections_transformation.md), [`YOLO-World Model`](yolo_world_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Anthropic Claude`](anthropic_claude.md), [`Camera Focus`](camera_focus.md), [`Track Class Lock`](track_class_lock.md), [`SmolVLM2`](smol_vlm2.md), [`Label Visualization`](label_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Text Display`](text_display.md), [`Qwen-VL`](qwen_vl.md), [`Velocity`](velocity.md), [`Gaze Detection`](gaze_detection.md), [`CSV Formatter`](csv_formatter.md), [`LMM`](lmm.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Line Counter`](line_counter.md), [`Qwen3-VL`](qwen3_vl.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemma API`](google_gemma_api.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Halo Visualization`](halo_visualization.md), [`Event Writer`](event_writer.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Property Definition`](property_definition.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Identify Outliers`](identify_outliers.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Clip Comparison`](clip_comparison.md), [`SIFT Comparison`](sift_comparison.md), [`Time in Zone`](timein_zone.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Slack Notification`](slack_notification.md), [`OpenRouter`](open_router.md), [`Detection Event Log`](detection_event_log.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Dynamic Zone`](dynamic_zone.md), [`Google Gemma`](google_gemma.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`GLM-OCR`](glmocr.md), [`Icon Visualization`](icon_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`QR Code Generator`](qr_code_generator.md), [`Path Deviation`](path_deviation.md), [`MQTT Writer`](mqtt_writer.md), [`Object Detection Model`](object_detection_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Dot Visualization`](dot_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Seg Preview`](seg_preview.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`SIFT`](sift.md), [`Local File Sink`](local_file_sink.md), [`Triangle Visualization`](triangle_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Polygon Visualization`](polygon_visualization.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Data Aggregator`](data_aggregator.md), [`Rate Limiter`](rate_limiter.md), [`Google Gemini`](google_gemini.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`LMM For Classification`](lmm_for_classification.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Stack`](image_stack.md), [`Email Notification`](email_notification.md), [`Mask Visualization`](mask_visualization.md), [`Distance Measurement`](distance_measurement.md), [`Barcode Detection`](barcode_detection.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Keypoint Visualization`](keypoint_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Overlap Filter`](overlap_filter.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Image Contours`](image_contours.md), [`Byte Tracker`](byte_tracker.md), [`SAM 3`](sam3.md), [`Motion Detection`](motion_detection.md), [`Current Time`](current_time.md), [`Corner Visualization`](corner_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Circle Visualization`](circle_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Background Color Visualization`](background_color_visualization.md), [`Line Counter`](line_counter.md), [`Template Matching`](template_matching.md), [`Morphological Transformation`](morphological_transformation.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Delta Filter`](delta_filter.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Size Measurement`](size_measurement.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Trace Visualization`](trace_visualization.md), [`JSON Parser`](json_parser.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Image Blur`](image_blur.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`QR Code Detection`](qr_code_detection.md), [`Camera Focus`](camera_focus.md), [`SORT Tracker`](sort_tracker.md), [`VLM As Detector`](vlm_as_detector.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Detections Stitch`](detections_stitch.md), [`Color Visualization`](color_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Buffer`](buffer.md), [`Cache Set`](cache_set.md), [`Time in Zone`](timein_zone.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`OpenAI`](open_ai.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Detection Offset`](detection_offset.md), [`Dominant Color`](dominant_color.md), [`CogVLM`](cog_vlm.md), [`Detections Consensus`](detections_consensus.md), [`Object Detection Model`](object_detection_model.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Dynamic Crop`](dynamic_crop.md), [`Path Deviation`](path_deviation.md), [`Byte Tracker`](byte_tracker.md), [`Expression`](expression.md), [`Detections Combine`](detections_combine.md), [`Continue If`](continue_if.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`SAM 3`](sam3.md), [`Cache Get`](cache_get.md), [`OpenAI`](open_ai.md), [`OCR Model`](ocr_model.md), [`SIFT Comparison`](sift_comparison.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`Halo Visualization`](halo_visualization.md), [`Image Threshold`](image_threshold.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Stitch Images`](stitch_images.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`VLM As Classifier`](vlm_as_classifier.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemini`](google_gemini.md), [`Byte Tracker`](byte_tracker.md), [`Webhook Sink`](webhook_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Detections Merge`](detections_merge.md), [`Google Gemini`](google_gemini.md), [`EasyOCR`](easy_ocr.md), [`Dimension Collapse`](dimension_collapse.md), [`SAM 3`](sam3.md), [`Time in Zone`](timein_zone.md), [`OpenAI`](open_ai.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`VLM As Detector`](vlm_as_detector.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Identify Changes`](identify_changes.md), [`Polygon Visualization`](polygon_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Detections Filter`](detections_filter.md), [`Background Subtraction`](background_subtraction.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Email Notification`](email_notification.md), [`Image Slicer`](image_slicer.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Depth Estimation`](depth_estimation.md), [`Pixel Color Count`](pixel_color_count.md), [`Qwen3.5`](qwen3.5.md), [`Cosine Similarity`](cosine_similarity.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Camera Calibration`](camera_calibration.md), [`Moondream2`](moondream2.md), [`Grid Visualization`](grid_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Inner Workflow`](inner_workflow.md), [`S3 Sink`](s3_sink.md), [`Image Slicer`](image_slicer.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Relative Static Crop`](relative_static_crop.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Object Detection Model`](object_detection_model.md)
    - outputs: [`Template Matching`](template_matching.md), [`Morphological Transformation`](morphological_transformation.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Blur Visualization`](blur_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`OpenAI`](open_ai.md), [`YOLO-World Model`](yolo_world_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Anthropic Claude`](anthropic_claude.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Size Measurement`](size_measurement.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Trace Visualization`](trace_visualization.md), [`Label Visualization`](label_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Text Display`](text_display.md), [`Qwen-VL`](qwen_vl.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Blur`](image_blur.md), [`Gaze Detection`](gaze_detection.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`LMM`](lmm.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Line Counter`](line_counter.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Detections Stitch`](detections_stitch.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemma API`](google_gemma_api.md), [`Halo Visualization`](halo_visualization.md), [`Color Visualization`](color_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Morphological Transformation`](morphological_transformation.md), [`Event Writer`](event_writer.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Cache Set`](cache_set.md), [`Time in Zone`](timein_zone.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`OpenAI`](open_ai.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`CogVLM`](cog_vlm.md), [`Detections Consensus`](detections_consensus.md), [`Object Detection Model`](object_detection_model.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Path Deviation`](path_deviation.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`SAM 3`](sam3.md), [`Cache Get`](cache_get.md), [`OpenAI`](open_ai.md), [`Time in Zone`](timein_zone.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Slack Notification`](slack_notification.md), [`OpenRouter`](open_router.md), [`SIFT Comparison`](sift_comparison.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Dynamic Zone`](dynamic_zone.md), [`Google Gemma`](google_gemma.md), [`Halo Visualization`](halo_visualization.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`GLM-OCR`](glmocr.md), [`Image Threshold`](image_threshold.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Icon Visualization`](icon_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Google Gemini`](google_gemini.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Webhook Sink`](webhook_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`QR Code Generator`](qr_code_generator.md), [`Path Deviation`](path_deviation.md), [`MQTT Writer`](mqtt_writer.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Dot Visualization`](dot_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Seg Preview`](seg_preview.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemini`](google_gemini.md), [`Local File Sink`](local_file_sink.md), [`SAM 3`](sam3.md), [`Triangle Visualization`](triangle_visualization.md), [`Time in Zone`](timein_zone.md), [`Contrast Equalization`](contrast_equalization.md), [`Polygon Visualization`](polygon_visualization.md), [`OpenAI`](open_ai.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Google Gemini`](google_gemini.md), [`LMM For Classification`](lmm_for_classification.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Stack`](image_stack.md), [`Email Notification`](email_notification.md), [`Polygon Visualization`](polygon_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Distance Measurement`](distance_measurement.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Keypoint Visualization`](keypoint_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Email Notification`](email_notification.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`SAM 3`](sam3.md), [`Depth Estimation`](depth_estimation.md), [`Pixel Color Count`](pixel_color_count.md), [`Motion Detection`](motion_detection.md), [`Current Time`](current_time.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Corner Visualization`](corner_visualization.md), [`Moondream2`](moondream2.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`S3 Sink`](s3_sink.md), [`Circle Visualization`](circle_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Object Detection Model`](object_detection_model.md), [`Background Color Visualization`](background_color_visualization.md), [`Line Counter`](line_counter.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Roboflow Vision Events` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `input_image` (*[`image`](../kinds/image.md)*): The original input image. Uploaded to the Vision Events API and used as the base image for detection annotations..
        - `output_image` (*[`image`](../kinds/image.md)*): An optional output/visualized image (e.g., from a visualization block). Displayed as the primary image in the Vision Events dashboard..
        - `predictions` (*Union[[`classification_prediction`](../kinds/classification_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md)]*): Optional model predictions to include as detection annotations on the input image. Supports object detection, instance segmentation, keypoint detection, and classification predictions..
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


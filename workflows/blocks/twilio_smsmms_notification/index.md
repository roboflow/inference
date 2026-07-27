
# Twilio SMS/MMS Notification



??? "Class: `TwilioSMSNotificationBlockV2`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/sinks/twilio/sms/v2.py">inference.core.workflows.core_steps.sinks.twilio.sms.v2.TwilioSMSNotificationBlockV2</a>
    



The **Twilio SMS/MMS Notification** block allows users to send text and multimedia messages as part of a workflow.

### SMS Provider Options

This block supports two SMS delivery methods via a dropdown selector:

1. **Roboflow Managed API Key (Default)** - No Twilio configuration needed. Messages are sent through Roboflow's proxy service:
   * **Simplified setup** - just provide message, recipient, and optional media
   * **Secure** - your workflow API key is used for authentication
   * **No Twilio account required**
   * **Pricing:** US/Canada: 30 messages per credit. International: 10 messages per credit. (SMS and MMS priced the same)

2. **Custom Twilio** - Use your own Twilio account:
   * Full control over message delivery
   * Requires Twilio credentials (Account SID, Auth Token, Phone Number)
   * You pay Twilio directly for messaging

### Message Content

* **Receiver Number:** Phone number to receive the message (must be in E.164 format, e.g., +15551234567)

* **Message:** The body content of the SMS/MMS. **Message can be parametrised with data generated during workflow run. See *Dynamic Parameters* section.**

* **Media URL (Optional):** For MMS messages, provide image URL(s) or image outputs from visualization blocks

### Dynamic Parameters

Message content can be parametrised with Workflow execution outcomes. Example:

```
message = "Alert! Detected {{ '{{' }} $parameters.num_detections {{ '}}' }} objects"
```

Message parameters are set via `message_parameters`:

```
message_parameters = {
    "num_detections": "$steps.model.predictions"
}
```

Transform data using `message_parameters_operations`:

```
message_parameters_operations = {
    "predictions": [
        {"type": "DetectionsPropertyExtract", "property_name": "class_name"}
    ]
}
```

### MMS Support

Send images with your message by providing `media_url`:

* **Image URLs**: Provide publicly accessible image URLs as a string or list
* **Workflow Images**: Reference image outputs from visualization blocks  
* **Base64 Images**: Images are automatically converted for transmission

Example:

```
media_url = "$steps.bounding_box_visualization.image"
```

Or multiple images:

```
media_url = ["https://example.com/image1.jpg", "https://example.com/image2.jpg"]
```

**Note:** MMS is primarily supported in US/Canada. International MMS availability varies by carrier.

### Using Custom Twilio

To use your own Twilio account, select "Custom Twilio" and configure:

* `twilio_account_sid` - Your Twilio Account SID from the [Twilio Console](https://twilio.com/console)
* `twilio_auth_token` - Your Twilio Auth Token  
* `sender_number` - Your Twilio phone number (must be in E.164 format)

### Cooldown

The block accepts `cooldown_seconds` (defaults to `5` seconds) to prevent notification bursts. Set `0` for no cooldown.

During cooldown, the `throttling_status` output is set to `True` and no message is sent.

!!! warning "Cooldown limitations"
    Cooldown is limited to video processing. Using this block in HTTP service workflows 
    (Roboflow Hosted API, Dedicated Deployment) has no cooldown effect for HTTP requests.

### Async Execution

Set `fire_and_forget=True` to send messages in the background, allowing the Workflow to proceed.  
With async mode, `error_status` is always `False`. **Set `fire_and_forget=False` for debugging.**

### Disabling Notifications

Set `disable_sink` flag to manually disable the notifier block via Workflow input.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/twilio_sms_notification@v2`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `sms_provider` | `str` | Choose SMS delivery method: use Roboflow's managed service or configure your own Twilio account.. | ❌ |
| `receiver_number` | `str` | Phone number to receive the message (E.164 format, e.g., +15551234567). | ✅ |
| `message` | `str` | Content of the message to be sent.. | ❌ |
| `message_parameters` | `Dict[str, Union[bool, float, int, str]]` | Data to be used inside the message.. | ✅ |
| `message_parameters_operations` | `Dict[str, List[Union[ClassificationPropertyExtract, ConvertDictionaryToJSON, ConvertImageToBase64, ConvertImageToJPEG, DetectionsFilter, DetectionsOffset, DetectionsPropertyExtract, DetectionsRename, DetectionsSelection, DetectionsShift, DetectionsToDictionary, Divide, ExtractDetectionProperty, ExtractFrameMetadata, ExtractImageProperty, LookupTable, Multiply, NumberRound, NumericSequenceAggregate, PickDetectionsByParentClass, RandomNumber, SequenceAggregate, SequenceApply, SequenceElementsCount, SequenceLength, SequenceMap, SortDetections, StringMatches, StringSubSequence, StringToLowerCase, StringToUpperCase, TimestampToISOFormat, ToBoolean, ToNumber, ToString]]]` | Preprocessing operations to be performed on message parameters.. | ❌ |
| `media_url` | `Optional[List[str], str]` | Optional media URL(s) for MMS. Provide publicly accessible image URLs or image outputs from workflow blocks.. | ✅ |
| `twilio_account_sid` | `str` | Twilio Account SID from the Twilio Console.. | ✅ |
| `twilio_auth_token` | `str` | Twilio Auth Token from the Twilio Console.. | ✅ |
| `sender_number` | `str` | Sender phone number (E.164 format, e.g., +15551234567). | ✅ |
| `fire_and_forget` | `bool` | Boolean flag to run the block asynchronously (True) for faster workflows or synchronously (False) for debugging and error handling.. | ✅ |
| `disable_sink` | `bool` | Boolean flag to disable block execution.. | ✅ |
| `cooldown_seconds` | `int` | Number of seconds until a follow-up notification can be sent.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-cloud-off-outline:{ style="color: #546e7a" } `requires_internet` — air-gapped / offline deployments
:   This block depends on a service that is not reachable from fully offline / air-gapped deployments.

:material-alert-circle-outline:{ style="color: #f57c00" } `soft` — runtime `hosted_serverless`, `dedicated_deployment`; execution `remote`
:   Cooldown / rate-limit timer is stored in process memory. With remote step execution on stateless or multi-replica HTTP runtimes each request gets a fresh worker, so cooldown does not throttle. Cooldown only behaves as documented with local step execution inside an InferencePipeline.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Twilio SMS/MMS Notification` in version `v2`.

    - inputs: [`Crop Visualization`](crop_visualization.md), [`Image Slicer`](image_slicer.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Google Gemini`](google_gemini.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Time in Zone`](timein_zone.md), [`Velocity`](velocity.md), [`Image Slicer`](image_slicer.md), [`Template Matching`](template_matching.md), [`Pixel Color Count`](pixel_color_count.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SIFT Comparison`](sift_comparison.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Morphological Transformation`](morphological_transformation.md), [`Background Subtraction`](background_subtraction.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Ellipse Visualization`](ellipse_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SAM 3`](sam3.md), [`Grid Visualization`](grid_visualization.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Google Gemma`](google_gemma.md), [`Byte Tracker`](byte_tracker.md), [`Overlap Filter`](overlap_filter.md), [`OpenAI`](open_ai.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Path Deviation`](path_deviation.md), [`Depth Estimation`](depth_estimation.md), [`Image Preprocessing`](image_preprocessing.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Detections Stabilizer`](detections_stabilizer.md), [`QR Code Detection`](qr_code_detection.md), [`Stitch Images`](stitch_images.md), [`Google Vision OCR`](google_vision_ocr.md), [`OpenAI`](open_ai.md), [`QR Code Generator`](qr_code_generator.md), [`Florence-2 Model`](florence2_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Detection Offset`](detection_offset.md), [`Cache Set`](cache_set.md), [`Contrast Equalization`](contrast_equalization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Clip Comparison`](clip_comparison.md), [`Pixelate Visualization`](pixelate_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Dot Visualization`](dot_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Byte Tracker`](byte_tracker.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`PP-OCR`](ppocr.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Stack`](image_stack.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Image Threshold`](image_threshold.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`JSON Parser`](json_parser.md), [`S3 Sink`](s3_sink.md), [`MQTT Writer`](mqtt_writer.md), [`Qwen-VL`](qwen_vl.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Detections Consensus`](detections_consensus.md), [`SIFT`](sift.md), [`Nearest Neighbor Detection Match`](nearest_neighbor_detection_match.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`Time in Zone`](timein_zone.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Dimension Collapse`](dimension_collapse.md), [`Trace Visualization`](trace_visualization.md), [`Seg Preview`](seg_preview.md), [`EasyOCR`](easy_ocr.md), [`Camera Focus`](camera_focus.md), [`Google Gemma API`](google_gemma_api.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Property Definition`](property_definition.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Distance Measurement`](distance_measurement.md), [`Data Aggregator`](data_aggregator.md), [`Blur Visualization`](blur_visualization.md), [`Path Deviation`](path_deviation.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Barcode Detection`](barcode_detection.md), [`Moondream2`](moondream2.md), [`Clip Comparison`](clip_comparison.md), [`Dynamic Zone`](dynamic_zone.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Email Notification`](email_notification.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Rate Limiter`](rate_limiter.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Contrast Enhancement`](contrast_enhancement.md), [`GLM-OCR`](glmocr.md), [`Byte Tracker`](byte_tracker.md), [`Camera Focus`](camera_focus.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Label Visualization`](label_visualization.md), [`Buffer`](buffer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Detections Merge`](detections_merge.md), [`Line Counter`](line_counter.md), [`Detections Filter`](detections_filter.md), [`Anthropic Claude`](anthropic_claude.md), [`Detection Event Log`](detection_event_log.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Dynamic Crop`](dynamic_crop.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Detections Transformation`](detections_transformation.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Overlap Analysis`](overlap_analysis.md), [`Current Time`](current_time.md), [`Expression`](expression.md), [`Cache Get`](cache_get.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Line Counter Visualization`](line_counter_visualization.md), [`OpenAI`](open_ai.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Object Detection Model`](object_detection_model.md), [`Cosmos 3`](cosmos3.md), [`Motion Detection`](motion_detection.md), [`Cosine Similarity`](cosine_similarity.md), [`Webhook Sink`](webhook_sink.md), [`VLM As Detector`](vlm_as_detector.md), [`Size Measurement`](size_measurement.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SAM 3`](sam3.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Qwen3.5`](qwen3.5.md), [`Camera Calibration`](camera_calibration.md), [`Qwen3-VL`](qwen3_vl.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`PLC Writer`](plc_writer.md), [`CSV Formatter`](csv_formatter.md), [`Detections Stitch`](detections_stitch.md), [`Track Class Lock`](track_class_lock.md), [`Triangle Visualization`](triangle_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Icon Visualization`](icon_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`VLM As Detector`](vlm_as_detector.md), [`OCR Model`](ocr_model.md), [`SAM 3`](sam3.md), [`GeoTag Detection`](geo_tag_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Halo Visualization`](halo_visualization.md), [`OpenRouter`](open_router.md), [`Frame Delay`](frame_delay.md), [`Local File Sink`](local_file_sink.md), [`Text Display`](text_display.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`LMM`](lmm.md), [`Image Blur`](image_blur.md), [`Gaze Detection`](gaze_detection.md), [`Inner Workflow`](inner_workflow.md), [`Color Visualization`](color_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Continue If`](continue_if.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`Slack Notification`](slack_notification.md), [`Event Writer`](event_writer.md), [`Identify Outliers`](identify_outliers.md), [`Google Gemini`](google_gemini.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Perspective Correction`](perspective_correction.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Florence-2 Model`](florence2_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Dominant Color`](dominant_color.md), [`Delta Filter`](delta_filter.md), [`Mask Visualization`](mask_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Switch Case`](switch_case.md), [`Halo Visualization`](halo_visualization.md), [`SmolVLM2`](smol_vlm2.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detections Combine`](detections_combine.md), [`CogVLM`](cog_vlm.md), [`Circle Visualization`](circle_visualization.md), [`Image Contours`](image_contours.md), [`Line Counter`](line_counter.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`YOLO-World Model`](yolo_world_model.md), [`Time in Zone`](timein_zone.md), [`Background Color Visualization`](background_color_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`PLC Reader`](plc_reader.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Email Notification`](email_notification.md), [`Identify Changes`](identify_changes.md)
    - outputs: [`Crop Visualization`](crop_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Time in Zone`](timein_zone.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Template Matching`](template_matching.md), [`Cache Get`](cache_get.md), [`Current Time`](current_time.md), [`Pixel Color Count`](pixel_color_count.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Line Counter Visualization`](line_counter_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Ellipse Visualization`](ellipse_visualization.md), [`OpenAI`](open_ai.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`SAM 3`](sam3.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Google Gemma`](google_gemma.md), [`Morphological Transformation`](morphological_transformation.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Object Detection Model`](object_detection_model.md), [`Motion Detection`](motion_detection.md), [`Cosmos 3`](cosmos3.md), [`OpenAI`](open_ai.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Webhook Sink`](webhook_sink.md), [`Path Deviation`](path_deviation.md), [`Size Measurement`](size_measurement.md), [`Depth Estimation`](depth_estimation.md), [`Image Preprocessing`](image_preprocessing.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SAM 3`](sam3.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`Camera Calibration`](camera_calibration.md), [`Google Vision OCR`](google_vision_ocr.md), [`OpenAI`](open_ai.md), [`QR Code Generator`](qr_code_generator.md), [`Florence-2 Model`](florence2_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Cache Set`](cache_set.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`PLC Writer`](plc_writer.md), [`Detections Stitch`](detections_stitch.md), [`Triangle Visualization`](triangle_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Icon Visualization`](icon_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Stack`](image_stack.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`SAM 3`](sam3.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Halo Visualization`](halo_visualization.md), [`OpenRouter`](open_router.md), [`Local File Sink`](local_file_sink.md), [`Text Display`](text_display.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`LMM`](lmm.md), [`Image Threshold`](image_threshold.md), [`Gaze Detection`](gaze_detection.md), [`Color Visualization`](color_visualization.md), [`Image Blur`](image_blur.md), [`Corner Visualization`](corner_visualization.md), [`S3 Sink`](s3_sink.md), [`MQTT Writer`](mqtt_writer.md), [`Qwen-VL`](qwen_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Google Gemini`](google_gemini.md), [`Slack Notification`](slack_notification.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Event Writer`](event_writer.md), [`Detections Consensus`](detections_consensus.md), [`Nearest Neighbor Detection Match`](nearest_neighbor_detection_match.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemini`](google_gemini.md), [`Time in Zone`](timein_zone.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Trace Visualization`](trace_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Seg Preview`](seg_preview.md), [`Google Gemma API`](google_gemma_api.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Distance Measurement`](distance_measurement.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Blur Visualization`](blur_visualization.md), [`Path Deviation`](path_deviation.md), [`LMM For Classification`](lmm_for_classification.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Florence-2 Model`](florence2_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Moondream2`](moondream2.md), [`Clip Comparison`](clip_comparison.md), [`Dynamic Zone`](dynamic_zone.md), [`Mask Visualization`](mask_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Email Notification`](email_notification.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Halo Visualization`](halo_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`GLM-OCR`](glmocr.md), [`CogVLM`](cog_vlm.md), [`Circle Visualization`](circle_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Line Counter`](line_counter.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`YOLO-World Model`](yolo_world_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Label Visualization`](label_visualization.md), [`Time in Zone`](timein_zone.md), [`Background Color Visualization`](background_color_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Line Counter`](line_counter.md), [`Object Detection Model`](object_detection_model.md), [`Email Notification`](email_notification.md), [`Anthropic Claude`](anthropic_claude.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Twilio SMS/MMS Notification` in version `v2`  has.

???+ tip "Bindings"

    - input
    
        - `receiver_number` (*[`string`](../kinds/string.md)*): Phone number to receive the message (E.164 format, e.g., +15551234567).
        - `message_parameters` (*[`*`](../kinds/wildcard.md)*): Data to be used inside the message..
        - `media_url` (*Union[[`list_of_values`](../kinds/list_of_values.md), [`string`](../kinds/string.md), [`image`](../kinds/image.md)]*): Optional media URL(s) for MMS. Provide publicly accessible image URLs or image outputs from workflow blocks..
        - `twilio_account_sid` (*Union[[`string`](../kinds/string.md), [`secret`](../kinds/secret.md)]*): Twilio Account SID from the Twilio Console..
        - `twilio_auth_token` (*Union[[`string`](../kinds/string.md), [`secret`](../kinds/secret.md)]*): Twilio Auth Token from the Twilio Console..
        - `sender_number` (*[`string`](../kinds/string.md)*): Sender phone number (E.164 format, e.g., +15551234567).
        - `fire_and_forget` (*[`boolean`](../kinds/boolean.md)*): Boolean flag to run the block asynchronously (True) for faster workflows or synchronously (False) for debugging and error handling..
        - `disable_sink` (*[`boolean`](../kinds/boolean.md)*): Boolean flag to disable block execution..
        - `cooldown_seconds` (*[`integer`](../kinds/integer.md)*): Number of seconds until a follow-up notification can be sent..

    - output
    
        - `error_status` ([`boolean`](../kinds/boolean.md)): Boolean flag.
        - `throttling_status` ([`boolean`](../kinds/boolean.md)): Boolean flag.
        - `message` ([`string`](../kinds/string.md)): String value.



??? tip "Example JSON definition of step `Twilio SMS/MMS Notification` in version `v2`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/twilio_sms_notification@v2",
	    "sms_provider": "Roboflow Managed API Key",
	    "receiver_number": "+15551234567",
	    "message": "Alert! Detected {{ '{{' }} $parameters.num_detections {{ '}}' }} objects",
	    "message_parameters": {
	        "num_detections": "$steps.model.predictions",
	        "reference": "$inputs.reference_class_names"
	    },
	    "message_parameters_operations": {
	        "predictions": [
	            {
	                "property_name": "class_name",
	                "type": "DetectionsPropertyExtract"
	            }
	        ]
	    },
	    "media_url": "$steps.visualization.image",
	    "twilio_account_sid": "$inputs.twilio_account_sid",
	    "twilio_auth_token": "$inputs.twilio_auth_token",
	    "sender_number": "+15551234567",
	    "fire_and_forget": "$inputs.fire_and_forget",
	    "disable_sink": false,
	    "cooldown_seconds": "$inputs.cooldown_seconds"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    


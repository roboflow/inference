
# Webhook Sink



??? "Class: `WebhookSinkBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/sinks/webhook/v1.py">inference.core.workflows.core_steps.sinks.webhook.v1.WebhookSinkBlockV1</a>
    



The **Webhook Sink** block enables sending a data from Workflow into external APIs 
by sending HTTP requests containing workflow results.

## How This Block Works

It supports multiple HTTP methods 
(GET, POST, PUT) and can be configured to send:

* JSON payloads

* query parameters

* multipart-encoded files 

This block is designed to provide flexibility for integrating workflows with remote systems 
for data exchange, notifications, or other integrations.

### Setting Query Parameters
You can easily set query parameters for your request:

```
query_parameters = {
    "api_key": "$inputs.api_key",
}
```

will send request into the following URL: `https://your-host/some/resource?api_key=<API_KEY_VALUE>`

### Setting headers
Setting headers is as easy as setting query parameters:

```
headers = {
    "api_key": "$inputs.api_key",
}
```

### Sending JSON payloads

You can set the body of your message to be JSON document that you construct specifying `json_payload` 
and `json_payload_operations` properties. `json_payload` works similarly to `query_parameters` and 
`headers`, but you can optionally apply UQL operations on top of JSON body elements.

Let's assume that you want to send number of bounding boxes predicted by object detection model
in body field named `detections_number`, then you need to specify configuration similar to the 
following:

```
json_payload = {
    "detections_number": "$steps.model.predictions",
}
json_payload_operations = {
    "detections_number": [{"type": "SequenceLength"}]
}
```

### Multipart-Encoded Files in POST requests

Your endpoint may also accept multipart requests. You can form them in a way which is similar to 
JSON payloads - setting `multi_part_encoded_files` and `multi_part_encoded_files_operations`.

Let's assume you want to send the image in the request, then your configuration may be the following:

```
multi_part_encoded_files = {
    "image": "$inputs.image",
}
multi_part_encoded_files_operations = {
    "image": [{"type": "ConvertImageToJPEG"}]
}
```

### Cooldown

The block accepts `cooldown_seconds` (which **defaults to `5` seconds**) to prevent unintended bursts of 
notifications. Please adjust it according to your needs, setting `0` indicate no cooldown. 

During cooldown period, consecutive runs of the step will cause `throttling_status` output to be set `True`
and no notification will be sent.

!!! warning "Cooldown limitations"

    Current implementation of cooldown is limited to video processing - using this block in context of a 
    Workflow that is run behind HTTP service (Roboflow Hosted API, Dedicated Deployment or self-hosted 
    `inference` server) will have no effect for processing HTTP requests.  
    

### Async execution

Configure the `fire_and_forget` property. Set it to True if you want the request to be sent in the background, 
allowing the Workflow to proceed without waiting on data to be sent. In this case you will not be able to rely on 
`error_status` output which will always be set to `False`, so we **recommend setting the `fire_and_forget=False` for
debugging purposes**.

### Disabling notifications based on runtime parameter

Sometimes it would be convenient to manually disable the **Webhook sink** block. This is possible 
setting `disable_sink` flag to hold reference to Workflow input. with such setup, caller would be
able to disable the sink when needed sending agreed input parameter.

## Common Use Cases

- Use this block to [purpose based on block type]

## Connecting to Other Blocks

The outputs from this block can be connected to other blocks in your workflow.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/webhook_sink@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `url` | `str` | URL of the resource to make request. | ✅ |
| `method` | `str` | HTTP method to be used. | ❌ |
| `query_parameters` | `Dict[str, Union[List[Union[bool, float, int, str]], bool, float, int, str]]` | Request query parameters. | ✅ |
| `headers` | `Dict[str, Union[bool, float, int, str]]` | Request headers. | ✅ |
| `json_payload` | `Dict[str, Union[Dict[Any, Any], List[Any], bool, float, int, str]]` | Fields to put into JSON payload. | ✅ |
| `json_payload_operations` | `Dict[str, List[Union[ClassificationPropertyExtract, ConvertDictionaryToJSON, ConvertImageToBase64, ConvertImageToJPEG, DetectionsFilter, DetectionsOffset, DetectionsPropertyExtract, DetectionsRename, DetectionsSelection, DetectionsShift, DetectionsToDictionary, Divide, ExtractDetectionProperty, ExtractFrameMetadata, ExtractImageProperty, LookupTable, Multiply, NumberRound, NumericSequenceAggregate, PickDetectionsByParentClass, RandomNumber, SequenceAggregate, SequenceApply, SequenceElementsCount, SequenceLength, SequenceMap, SortDetections, StringMatches, StringSubSequence, StringToLowerCase, StringToUpperCase, TimestampToISOFormat, ToBoolean, ToNumber, ToString]]]` | UQL definitions of operations to be performed on defined data w.r.t. each value of `json_payload` parameter. | ❌ |
| `multi_part_encoded_files` | `Dict[str, Union[Dict[Any, Any], List[Any], bool, float, int, str]]` | Data to POST as Multipart-Encoded File. | ✅ |
| `multi_part_encoded_files_operations` | `Dict[str, List[Union[ClassificationPropertyExtract, ConvertDictionaryToJSON, ConvertImageToBase64, ConvertImageToJPEG, DetectionsFilter, DetectionsOffset, DetectionsPropertyExtract, DetectionsRename, DetectionsSelection, DetectionsShift, DetectionsToDictionary, Divide, ExtractDetectionProperty, ExtractFrameMetadata, ExtractImageProperty, LookupTable, Multiply, NumberRound, NumericSequenceAggregate, PickDetectionsByParentClass, RandomNumber, SequenceAggregate, SequenceApply, SequenceElementsCount, SequenceLength, SequenceMap, SortDetections, StringMatches, StringSubSequence, StringToLowerCase, StringToUpperCase, TimestampToISOFormat, ToBoolean, ToNumber, ToString]]]` | UQL definitions of operations to be performed on defined data w.r.t. each value of `multi_part_encoded_files` parameter. | ❌ |
| `form_data` | `Dict[str, Union[Dict[Any, Any], List[Any], bool, float, int, str]]` | Fields to put into form-data. | ✅ |
| `form_data_operations` | `Dict[str, List[Union[ClassificationPropertyExtract, ConvertDictionaryToJSON, ConvertImageToBase64, ConvertImageToJPEG, DetectionsFilter, DetectionsOffset, DetectionsPropertyExtract, DetectionsRename, DetectionsSelection, DetectionsShift, DetectionsToDictionary, Divide, ExtractDetectionProperty, ExtractFrameMetadata, ExtractImageProperty, LookupTable, Multiply, NumberRound, NumericSequenceAggregate, PickDetectionsByParentClass, RandomNumber, SequenceAggregate, SequenceApply, SequenceElementsCount, SequenceLength, SequenceMap, SortDetections, StringMatches, StringSubSequence, StringToLowerCase, StringToUpperCase, TimestampToISOFormat, ToBoolean, ToNumber, ToString]]]` | UQL definitions of operations to be performed on defined data w.r.t. each value of `form_data` parameter. | ❌ |
| `request_timeout` | `int` | Number of seconds to wait for remote API response. | ✅ |
| `fire_and_forget` | `bool` | Boolean flag to run the block asynchronously (True) for faster workflows or  synchronously (False) for debugging and error handling.. | ✅ |
| `disable_sink` | `bool` | Boolean flag to disable block execution.. | ✅ |
| `cooldown_seconds` | `int` | Number of seconds to wait until follow-up notification can be sent.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-cloud-off-outline:{ style="color: #546e7a" } `requires_internet` — air-gapped / offline deployments
:   This block depends on a service that is not reachable from fully offline / air-gapped deployments.

:material-alert-circle-outline:{ style="color: #f57c00" } `soft` — runtime `hosted_serverless`, `dedicated_deployment`; execution `remote`
:   Cooldown / rate-limit timer is stored in process memory. With remote step execution on stateless or multi-replica HTTP runtimes each request gets a fresh worker, so cooldown does not throttle. Cooldown only behaves as documented with local step execution inside an InferencePipeline.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Webhook Sink` in version `v1`.

    - inputs: [`PLC Writer`](plc_writer.md), [`Image Blur`](image_blur.md), [`Track Class Lock`](track_class_lock.md), [`Byte Tracker`](byte_tracker.md), [`Mask Edge Snap`](mask_edge_snap.md), [`VLM As Detector`](vlm_as_detector.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Image Slicer`](image_slicer.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Motion Detection`](motion_detection.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Clip Comparison`](clip_comparison.md), [`Buffer`](buffer.md), [`Label Visualization`](label_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`S3 Sink`](s3_sink.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Velocity`](velocity.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`CSV Formatter`](csv_formatter.md), [`Background Subtraction`](background_subtraction.md), [`Corner Visualization`](corner_visualization.md), [`Detection Offset`](detection_offset.md), [`JSON Parser`](json_parser.md), [`Property Definition`](property_definition.md), [`Google Gemini`](google_gemini.md), [`Seg Preview`](seg_preview.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`LMM For Classification`](lmm_for_classification.md), [`PP-OCR`](ppocr.md), [`SIFT Comparison`](sift_comparison.md), [`Object Detection Model`](object_detection_model.md), [`Color Visualization`](color_visualization.md), [`Inner Workflow`](inner_workflow.md), [`Local File Sink`](local_file_sink.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Dynamic Zone`](dynamic_zone.md), [`Google Gemma API`](google_gemma_api.md), [`Detections Filter`](detections_filter.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Icon Visualization`](icon_visualization.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Detections Transformation`](detections_transformation.md), [`Rich Label Visualization`](rich_label_visualization.md), [`Identify Changes`](identify_changes.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Expression`](expression.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Google Gemini`](google_gemini.md), [`Detections Stitch`](detections_stitch.md), [`Event Writer`](event_writer.md), [`CogVLM`](cog_vlm.md), [`Time in Zone`](timein_zone.md), [`Email Notification`](email_notification.md), [`Current Time`](current_time.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Perspective Correction`](perspective_correction.md), [`Cosine Similarity`](cosine_similarity.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Overlap Filter`](overlap_filter.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Cache Get`](cache_get.md), [`OpenRouter`](open_router.md), [`Detections Consensus`](detections_consensus.md), [`OCR Model`](ocr_model.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Time in Zone`](timein_zone.md), [`Relative Static Crop`](relative_static_crop.md), [`SAM 3`](sam3.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`GLM-OCR`](glmocr.md), [`Florence-2 Model`](florence2_model.md), [`Background Color Visualization`](background_color_visualization.md), [`Qwen3.5`](qwen3.5.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Threshold`](image_threshold.md), [`Camera Focus`](camera_focus.md), [`Line Counter`](line_counter.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Circle Visualization`](circle_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Detections Merge`](detections_merge.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Cosmos 3`](cosmos3.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`MQTT Writer`](mqtt_writer.md), [`SAM 3`](sam3.md), [`Line Counter`](line_counter.md), [`Image Stack`](image_stack.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Florence-2 Model`](florence2_model.md), [`Grid Visualization`](grid_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Path Deviation`](path_deviation.md), [`Crop Visualization`](crop_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Webhook Sink`](webhook_sink.md), [`VLM As Classifier`](vlm_as_classifier.md), [`SAM 3 Interactive`](sam3_interactive.md), [`YOLO-World Model`](yolo_world_model.md), [`SIFT`](sift.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Slack Notification`](slack_notification.md), [`PLC Reader`](plc_reader.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Label Visualization`](label_visualization.md), [`Email Notification`](email_notification.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Overlap Analysis`](overlap_analysis.md), [`Pixel Color Count`](pixel_color_count.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Path Deviation`](path_deviation.md), [`Stitch Images`](stitch_images.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Distance Measurement`](distance_measurement.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Camera Calibration`](camera_calibration.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Camera Focus`](camera_focus.md), [`Delta Filter`](delta_filter.md), [`Trace Visualization`](trace_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Dominant Color`](dominant_color.md), [`Morphological Transformation`](morphological_transformation.md), [`Continue If`](continue_if.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Google Vision OCR`](google_vision_ocr.md), [`Google Gemma`](google_gemma.md), [`Polygon Visualization`](polygon_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Image Slicer`](image_slicer.md), [`OpenAI`](open_ai.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Object Detection Model`](object_detection_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Mask Visualization`](mask_visualization.md), [`Moondream2`](moondream2.md), [`Nearest Neighbor Detection Match`](nearest_neighbor_detection_match.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Byte Tracker`](byte_tracker.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`VLM As Detector`](vlm_as_detector.md), [`Blur Visualization`](blur_visualization.md), [`Frame Delay`](frame_delay.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Switch Case`](switch_case.md), [`Cache Set`](cache_set.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Barcode Detection`](barcode_detection.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Template Matching`](template_matching.md), [`QR Code Detection`](qr_code_detection.md), [`Halo Visualization`](halo_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`Dynamic Crop`](dynamic_crop.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`GeoTag Detection`](geo_tag_detection.md), [`Halo Visualization`](halo_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Time in Zone`](timein_zone.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Size Measurement`](size_measurement.md), [`LMM`](lmm.md), [`Depth Estimation`](depth_estimation.md), [`SIFT Comparison`](sift_comparison.md), [`Detections Combine`](detections_combine.md), [`Dot Visualization`](dot_visualization.md), [`EasyOCR`](easy_ocr.md), [`Rate Limiter`](rate_limiter.md), [`Google Gemini`](google_gemini.md), [`Data Aggregator`](data_aggregator.md), [`Image Contours`](image_contours.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Morphological Transformation`](morphological_transformation.md), [`SAM 3`](sam3.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Gemini`](google_gemini.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Anthropic Claude`](anthropic_claude.md), [`SORT Tracker`](sort_tracker.md), [`Image Preprocessing`](image_preprocessing.md), [`Qwen3-VL`](qwen3_vl.md), [`Gaze Detection`](gaze_detection.md), [`Anthropic Claude`](anthropic_claude.md), [`Text Display`](text_display.md), [`Identify Outliers`](identify_outliers.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`SmolVLM2`](smol_vlm2.md)
    - outputs: [`PLC Writer`](plc_writer.md), [`Image Blur`](image_blur.md), [`Path Deviation`](path_deviation.md), [`Crop Visualization`](crop_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Webhook Sink`](webhook_sink.md), [`Motion Detection`](motion_detection.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`YOLO-World Model`](yolo_world_model.md), [`Slack Notification`](slack_notification.md), [`Label Visualization`](label_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Label Visualization`](label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`S3 Sink`](s3_sink.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Email Notification`](email_notification.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Pixel Color Count`](pixel_color_count.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Path Deviation`](path_deviation.md), [`Corner Visualization`](corner_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Distance Measurement`](distance_measurement.md), [`Camera Calibration`](camera_calibration.md), [`Google Gemini`](google_gemini.md), [`Seg Preview`](seg_preview.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`LMM For Classification`](lmm_for_classification.md), [`SIFT Comparison`](sift_comparison.md), [`Trace Visualization`](trace_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Color Visualization`](color_visualization.md), [`Local File Sink`](local_file_sink.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Dynamic Zone`](dynamic_zone.md), [`Morphological Transformation`](morphological_transformation.md), [`Google Vision OCR`](google_vision_ocr.md), [`Google Gemma API`](google_gemma_api.md), [`Google Gemma`](google_gemma.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Polygon Visualization`](polygon_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Rich Label Visualization`](rich_label_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Keypoint Visualization`](keypoint_visualization.md), [`OpenAI`](open_ai.md), [`Object Detection Model`](object_detection_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Mask Visualization`](mask_visualization.md), [`Moondream2`](moondream2.md), [`Nearest Neighbor Detection Match`](nearest_neighbor_detection_match.md), [`OpenAI`](open_ai.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Blur Visualization`](blur_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Reference Path Visualization`](reference_path_visualization.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Gemini`](google_gemini.md), [`Detections Stitch`](detections_stitch.md), [`Event Writer`](event_writer.md), [`Cache Set`](cache_set.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`CogVLM`](cog_vlm.md), [`Time in Zone`](timein_zone.md), [`Email Notification`](email_notification.md), [`Current Time`](current_time.md), [`Template Matching`](template_matching.md), [`Perspective Correction`](perspective_correction.md), [`Halo Visualization`](halo_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Cache Get`](cache_get.md), [`OpenRouter`](open_router.md), [`Dynamic Crop`](dynamic_crop.md), [`Detections Consensus`](detections_consensus.md), [`Halo Visualization`](halo_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Time in Zone`](timein_zone.md), [`Time in Zone`](timein_zone.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Size Measurement`](size_measurement.md), [`LMM`](lmm.md), [`Depth Estimation`](depth_estimation.md), [`SAM 3`](sam3.md), [`Dot Visualization`](dot_visualization.md), [`GLM-OCR`](glmocr.md), [`Florence-2 Model`](florence2_model.md), [`Background Color Visualization`](background_color_visualization.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Contrast Equalization`](contrast_equalization.md), [`Morphological Transformation`](morphological_transformation.md), [`Image Threshold`](image_threshold.md), [`SAM 3`](sam3.md), [`Line Counter`](line_counter.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Circle Visualization`](circle_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Google Gemini`](google_gemini.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Cosmos 3`](cosmos3.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`MQTT Writer`](mqtt_writer.md), [`SAM 3`](sam3.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Preprocessing`](image_preprocessing.md), [`Gaze Detection`](gaze_detection.md), [`Line Counter`](line_counter.md), [`Anthropic Claude`](anthropic_claude.md), [`Text Display`](text_display.md), [`Image Stack`](image_stack.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Webhook Sink` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `url` (*[`string`](../kinds/string.md)*): URL of the resource to make request.
        - `query_parameters` (*Union[[`roboflow_model_id`](../kinds/roboflow_model_id.md), [`string`](../kinds/string.md), [`list_of_values`](../kinds/list_of_values.md), [`top_class`](../kinds/top_class.md), [`roboflow_api_key`](../kinds/roboflow_api_key.md), [`roboflow_project`](../kinds/roboflow_project.md), [`boolean`](../kinds/boolean.md), [`float`](../kinds/float.md), [`integer`](../kinds/integer.md), [`float_zero_to_one`](../kinds/float_zero_to_one.md)]*): Request query parameters.
        - `headers` (*Union[[`roboflow_model_id`](../kinds/roboflow_model_id.md), [`string`](../kinds/string.md), [`top_class`](../kinds/top_class.md), [`roboflow_api_key`](../kinds/roboflow_api_key.md), [`roboflow_project`](../kinds/roboflow_project.md), [`boolean`](../kinds/boolean.md), [`float`](../kinds/float.md), [`integer`](../kinds/integer.md), [`float_zero_to_one`](../kinds/float_zero_to_one.md)]*): Request headers.
        - `json_payload` (*[`*`](../kinds/wildcard.md)*): Fields to put into JSON payload.
        - `multi_part_encoded_files` (*[`*`](../kinds/wildcard.md)*): Data to POST as Multipart-Encoded File.
        - `form_data` (*[`*`](../kinds/wildcard.md)*): Fields to put into form-data.
        - `request_timeout` (*[`integer`](../kinds/integer.md)*): Number of seconds to wait for remote API response.
        - `fire_and_forget` (*[`boolean`](../kinds/boolean.md)*): Boolean flag to run the block asynchronously (True) for faster workflows or  synchronously (False) for debugging and error handling..
        - `disable_sink` (*[`boolean`](../kinds/boolean.md)*): Boolean flag to disable block execution..
        - `cooldown_seconds` (*[`integer`](../kinds/integer.md)*): Number of seconds to wait until follow-up notification can be sent..

    - output
    
        - `error_status` ([`boolean`](../kinds/boolean.md)): Boolean flag.
        - `throttling_status` ([`boolean`](../kinds/boolean.md)): Boolean flag.
        - `message` ([`string`](../kinds/string.md)): String value.



??? tip "Example JSON definition of step `Webhook Sink` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/webhook_sink@v1",
	    "url": "<block_does_not_provide_example>",
	    "method": "<block_does_not_provide_example>",
	    "query_parameters": {
	        "api_key": "$inputs.api_key"
	    },
	    "headers": {
	        "api_key": "$inputs.api_key"
	    },
	    "json_payload": {
	        "field": "$steps.model.predictions"
	    },
	    "json_payload_operations": {
	        "predictions": [
	            {
	                "property_name": "class_name",
	                "type": "DetectionsPropertyExtract"
	            }
	        ]
	    },
	    "multi_part_encoded_files": {
	        "image": "$steps.visualization.image"
	    },
	    "multi_part_encoded_files_operations": {
	        "predictions": [
	            {
	                "property_name": "class_name",
	                "type": "DetectionsPropertyExtract"
	            }
	        ]
	    },
	    "form_data": {
	        "field": "$inputs.field_value"
	    },
	    "form_data_operations": {
	        "predictions": [
	            {
	                "property_name": "class_name",
	                "type": "DetectionsPropertyExtract"
	            }
	        ]
	    },
	    "request_timeout": "$inputs.request_timeout",
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


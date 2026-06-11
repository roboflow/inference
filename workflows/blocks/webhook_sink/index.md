
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

    - inputs: [`Camera Focus`](camera_focus.md), [`Current Time`](current_time.md), [`Camera Focus`](camera_focus.md), [`Image Preprocessing`](image_preprocessing.md), [`Background Subtraction`](background_subtraction.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Color Visualization`](color_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Seg Preview`](seg_preview.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`JSON Parser`](json_parser.md), [`Cache Set`](cache_set.md), [`Google Vision OCR`](google_vision_ocr.md), [`SIFT Comparison`](sift_comparison.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Keypoint Visualization`](keypoint_visualization.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Camera Calibration`](camera_calibration.md), [`VLM As Detector`](vlm_as_detector.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Expression`](expression.md), [`LMM For Classification`](lmm_for_classification.md), [`Barcode Detection`](barcode_detection.md), [`Perspective Correction`](perspective_correction.md), [`Cache Get`](cache_get.md), [`Detection Offset`](detection_offset.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Clip Comparison`](clip_comparison.md), [`Qwen-VL`](qwen_vl.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Dot Visualization`](dot_visualization.md), [`OpenAI`](open_ai.md), [`Moondream2`](moondream2.md), [`Size Measurement`](size_measurement.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Dominant Color`](dominant_color.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`SIFT`](sift.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Dimension Collapse`](dimension_collapse.md), [`Email Notification`](email_notification.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Template Matching`](template_matching.md), [`Image Slicer`](image_slicer.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Text Display`](text_display.md), [`Distance Measurement`](distance_measurement.md), [`Time in Zone`](timein_zone.md), [`Blur Visualization`](blur_visualization.md), [`Qwen3-VL`](qwen3_vl.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`LMM`](lmm.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Circle Visualization`](circle_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Label Visualization`](label_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`OpenAI`](open_ai.md), [`CSV Formatter`](csv_formatter.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`CogVLM`](cog_vlm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Clip Comparison`](clip_comparison.md), [`Detections Merge`](detections_merge.md), [`Triangle Visualization`](triangle_visualization.md), [`Detections Consensus`](detections_consensus.md), [`Background Color Visualization`](background_color_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Delta Filter`](delta_filter.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`OpenAI`](open_ai.md), [`Email Notification`](email_notification.md), [`Slack Notification`](slack_notification.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detections Filter`](detections_filter.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Google Gemma API`](google_gemma_api.md), [`Event Writer`](event_writer.md), [`Google Gemini`](google_gemini.md), [`Rate Limiter`](rate_limiter.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Inner Workflow`](inner_workflow.md), [`Image Contours`](image_contours.md), [`SAM 3`](sam3.md), [`YOLO-World Model`](yolo_world_model.md), [`Crop Visualization`](crop_visualization.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Relative Static Crop`](relative_static_crop.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`EasyOCR`](easy_ocr.md), [`VLM As Detector`](vlm_as_detector.md), [`SAM 3`](sam3.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Qwen3.5`](qwen3.5.md), [`Google Gemma`](google_gemma.md), [`Google Gemini`](google_gemini.md), [`Identify Changes`](identify_changes.md), [`Image Blur`](image_blur.md), [`MQTT Writer`](mqtt_writer.md), [`Buffer`](buffer.md), [`Anthropic Claude`](anthropic_claude.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`SIFT Comparison`](sift_comparison.md), [`Detections Stitch`](detections_stitch.md), [`Overlap Analysis`](overlap_analysis.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`SAM 3`](sam3.md), [`Time in Zone`](timein_zone.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Contrast Equalization`](contrast_equalization.md), [`SORT Tracker`](sort_tracker.md), [`Image Stack`](image_stack.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Identify Outliers`](identify_outliers.md), [`Dynamic Zone`](dynamic_zone.md), [`Continue If`](continue_if.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`QR Code Detection`](qr_code_detection.md), [`Property Definition`](property_definition.md), [`Path Deviation`](path_deviation.md), [`Webhook Sink`](webhook_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Image Slicer`](image_slicer.md), [`Halo Visualization`](halo_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Local File Sink`](local_file_sink.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`SmolVLM2`](smol_vlm2.md), [`Path Deviation`](path_deviation.md), [`Depth Estimation`](depth_estimation.md), [`Byte Tracker`](byte_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Icon Visualization`](icon_visualization.md), [`Motion Detection`](motion_detection.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Anthropic Claude`](anthropic_claude.md), [`Cosine Similarity`](cosine_similarity.md), [`Grid Visualization`](grid_visualization.md), [`OpenRouter`](open_router.md), [`Anthropic Claude`](anthropic_claude.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Image Threshold`](image_threshold.md), [`Overlap Filter`](overlap_filter.md), [`Data Aggregator`](data_aggregator.md), [`Trace Visualization`](trace_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Morphological Transformation`](morphological_transformation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Detections Transformation`](detections_transformation.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Gaze Detection`](gaze_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Byte Tracker`](byte_tracker.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Stitch Images`](stitch_images.md), [`Florence-2 Model`](florence2_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Byte Tracker`](byte_tracker.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Corner Visualization`](corner_visualization.md), [`Detections Combine`](detections_combine.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`OCR Model`](ocr_model.md), [`Line Counter`](line_counter.md), [`S3 Sink`](s3_sink.md), [`GLM-OCR`](glmocr.md), [`Detection Event Log`](detection_event_log.md), [`OpenAI`](open_ai.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Velocity`](velocity.md)
    - outputs: [`Google Gemma API`](google_gemma_api.md), [`Event Writer`](event_writer.md), [`Google Gemini`](google_gemini.md), [`Current Time`](current_time.md), [`Ellipse Visualization`](ellipse_visualization.md), [`SAM 3`](sam3.md), [`Image Preprocessing`](image_preprocessing.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Color Visualization`](color_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`YOLO-World Model`](yolo_world_model.md), [`Seg Preview`](seg_preview.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Cache Set`](cache_set.md), [`Google Vision OCR`](google_vision_ocr.md), [`SIFT Comparison`](sift_comparison.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Object Detection Model`](object_detection_model.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`SAM 3`](sam3.md), [`Google Gemma`](google_gemma.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Blur`](image_blur.md), [`Keypoint Visualization`](keypoint_visualization.md), [`MQTT Writer`](mqtt_writer.md), [`Anthropic Claude`](anthropic_claude.md), [`Camera Calibration`](camera_calibration.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Detections Stitch`](detections_stitch.md), [`LMM For Classification`](lmm_for_classification.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Perspective Correction`](perspective_correction.md), [`Cache Get`](cache_get.md), [`Time in Zone`](timein_zone.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`SAM 3`](sam3.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Contrast Equalization`](contrast_equalization.md), [`Clip Comparison`](clip_comparison.md), [`Qwen-VL`](qwen_vl.md), [`Image Stack`](image_stack.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Dot Visualization`](dot_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`OpenAI`](open_ai.md), [`Moondream2`](moondream2.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Size Measurement`](size_measurement.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Path Deviation`](path_deviation.md), [`Webhook Sink`](webhook_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Halo Visualization`](halo_visualization.md), [`Email Notification`](email_notification.md), [`QR Code Generator`](qr_code_generator.md), [`Local File Sink`](local_file_sink.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Object Detection Model`](object_detection_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Path Deviation`](path_deviation.md), [`Template Matching`](template_matching.md), [`Depth Estimation`](depth_estimation.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Text Display`](text_display.md), [`Time in Zone`](timein_zone.md), [`Object Detection Model`](object_detection_model.md), [`Distance Measurement`](distance_measurement.md), [`Icon Visualization`](icon_visualization.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`OpenRouter`](open_router.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Anthropic Claude`](anthropic_claude.md), [`LMM`](lmm.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Image Threshold`](image_threshold.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Label Visualization`](label_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Morphological Transformation`](morphological_transformation.md), [`Morphological Transformation`](morphological_transformation.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Gaze Detection`](gaze_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`CogVLM`](cog_vlm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Florence-2 Model`](florence2_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Background Color Visualization`](background_color_visualization.md), [`Detections Consensus`](detections_consensus.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Corner Visualization`](corner_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Line Counter Visualization`](line_counter_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Halo Visualization`](halo_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Line Counter`](line_counter.md), [`OpenAI`](open_ai.md), [`Email Notification`](email_notification.md), [`Slack Notification`](slack_notification.md), [`S3 Sink`](s3_sink.md), [`GLM-OCR`](glmocr.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OPC UA Writer Sink`](opcua_writer_sink.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Webhook Sink` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `url` (*[`string`](../kinds/string.md)*): URL of the resource to make request.
        - `query_parameters` (*Union[[`top_class`](../kinds/top_class.md), [`roboflow_project`](../kinds/roboflow_project.md), [`roboflow_api_key`](../kinds/roboflow_api_key.md), [`float`](../kinds/float.md), [`boolean`](../kinds/boolean.md), [`string`](../kinds/string.md), [`list_of_values`](../kinds/list_of_values.md), [`integer`](../kinds/integer.md), [`float_zero_to_one`](../kinds/float_zero_to_one.md), [`roboflow_model_id`](../kinds/roboflow_model_id.md)]*): Request query parameters.
        - `headers` (*Union[[`top_class`](../kinds/top_class.md), [`roboflow_project`](../kinds/roboflow_project.md), [`roboflow_api_key`](../kinds/roboflow_api_key.md), [`float`](../kinds/float.md), [`boolean`](../kinds/boolean.md), [`string`](../kinds/string.md), [`integer`](../kinds/integer.md), [`float_zero_to_one`](../kinds/float_zero_to_one.md), [`roboflow_model_id`](../kinds/roboflow_model_id.md)]*): Request headers.
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



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

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Webhook Sink` in version `v1`.

    - inputs: [`Slack Notification`](slack_notification.md), [`Camera Focus`](camera_focus.md), [`Rate Limiter`](rate_limiter.md), [`Background Color Visualization`](background_color_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Detections Combine`](detections_combine.md), [`Corner Visualization`](corner_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Line Counter Visualization`](line_counter_visualization.md), [`S3 Sink`](s3_sink.md), [`Triangle Visualization`](triangle_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dot Visualization`](dot_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Trace Visualization`](trace_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Detections Transformation`](detections_transformation.md), [`Cache Set`](cache_set.md), [`Byte Tracker`](byte_tracker.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`OpenAI`](open_ai.md), [`Barcode Detection`](barcode_detection.md), [`Line Counter`](line_counter.md), [`Google Gemma API`](google_gemma_api.md), [`OpenAI`](open_ai.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Clip Comparison`](clip_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`CSV Formatter`](csv_formatter.md), [`Image Preprocessing`](image_preprocessing.md), [`Grid Visualization`](grid_visualization.md), [`Delta Filter`](delta_filter.md), [`Time in Zone`](timein_zone.md), [`Camera Calibration`](camera_calibration.md), [`Object Detection Model`](object_detection_model.md), [`OCR Model`](ocr_model.md), [`Depth Estimation`](depth_estimation.md), [`OpenAI`](open_ai.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Qwen3-VL`](qwen3_vl.md), [`SAM 3`](sam3.md), [`Seg Preview`](seg_preview.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Vision OCR`](google_vision_ocr.md), [`Path Deviation`](path_deviation.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Motion Detection`](motion_detection.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Dynamic Crop`](dynamic_crop.md), [`Inner Workflow`](inner_workflow.md), [`Image Contours`](image_contours.md), [`VLM As Detector`](vlm_as_detector.md), [`Camera Focus`](camera_focus.md), [`Anthropic Claude`](anthropic_claude.md), [`Identify Outliers`](identify_outliers.md), [`Template Matching`](template_matching.md), [`Mask Edge Snap`](mask_edge_snap.md), [`QR Code Detection`](qr_code_detection.md), [`SIFT`](sift.md), [`QR Code Generator`](qr_code_generator.md), [`YOLO-World Model`](yolo_world_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Distance Measurement`](distance_measurement.md), [`Byte Tracker`](byte_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Icon Visualization`](icon_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Time in Zone`](timein_zone.md), [`Path Deviation`](path_deviation.md), [`Halo Visualization`](halo_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Pixel Color Count`](pixel_color_count.md), [`OpenAI`](open_ai.md), [`Halo Visualization`](halo_visualization.md), [`Image Slicer`](image_slicer.md), [`Object Detection Model`](object_detection_model.md), [`Mask Area Measurement`](mask_area_measurement.md), [`LMM`](lmm.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Mask Visualization`](mask_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Google Gemini`](google_gemini.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Cache Get`](cache_get.md), [`VLM As Detector`](vlm_as_detector.md), [`Property Definition`](property_definition.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`SAM 3`](sam3.md), [`Dimension Collapse`](dimension_collapse.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`EasyOCR`](easy_ocr.md), [`Background Subtraction`](background_subtraction.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Detection Offset`](detection_offset.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Expression`](expression.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`CogVLM`](cog_vlm.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Data Aggregator`](data_aggregator.md), [`Relative Static Crop`](relative_static_crop.md), [`Velocity`](velocity.md), [`SIFT Comparison`](sift_comparison.md), [`Color Visualization`](color_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Moondream2`](moondream2.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Dynamic Zone`](dynamic_zone.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`SmolVLM2`](smol_vlm2.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Slicer`](image_slicer.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Cosine Similarity`](cosine_similarity.md), [`Text Display`](text_display.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Stitch Images`](stitch_images.md), [`Morphological Transformation`](morphological_transformation.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Line Counter`](line_counter.md), [`SORT Tracker`](sort_tracker.md), [`Detections Consensus`](detections_consensus.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Identify Changes`](identify_changes.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Label Visualization`](label_visualization.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Byte Tracker`](byte_tracker.md), [`Local File Sink`](local_file_sink.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Overlap Filter`](overlap_filter.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`SAM 3`](sam3.md), [`JSON Parser`](json_parser.md), [`Detections Merge`](detections_merge.md), [`Anthropic Claude`](anthropic_claude.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Clip Comparison`](clip_comparison.md), [`Email Notification`](email_notification.md), [`Detections Stitch`](detections_stitch.md), [`Keypoint Visualization`](keypoint_visualization.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Size Measurement`](size_measurement.md), [`Email Notification`](email_notification.md), [`Florence-2 Model`](florence2_model.md), [`Detections Filter`](detections_filter.md), [`GLM-OCR`](glmocr.md), [`Continue If`](continue_if.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemini`](google_gemini.md), [`Circle Visualization`](circle_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Image Threshold`](image_threshold.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Dominant Color`](dominant_color.md), [`Gaze Detection`](gaze_detection.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Blur`](image_blur.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Buffer`](buffer.md), [`Qwen 3.6 API`](qwen3.6_api.md)
    - outputs: [`Slack Notification`](slack_notification.md), [`Background Color Visualization`](background_color_visualization.md), [`Time in Zone`](timein_zone.md), [`Perspective Correction`](perspective_correction.md), [`SAM 3`](sam3.md), [`Corner Visualization`](corner_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Line Counter Visualization`](line_counter_visualization.md), [`S3 Sink`](s3_sink.md), [`Triangle Visualization`](triangle_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dot Visualization`](dot_visualization.md), [`CogVLM`](cog_vlm.md), [`Trace Visualization`](trace_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Cache Set`](cache_set.md), [`Color Visualization`](color_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`Moondream2`](moondream2.md), [`Line Counter`](line_counter.md), [`Google Gemma API`](google_gemma_api.md), [`OpenAI`](open_ai.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Dynamic Zone`](dynamic_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Morphological Transformation`](morphological_transformation.md), [`Clip Comparison`](clip_comparison.md), [`Image Preprocessing`](image_preprocessing.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Time in Zone`](timein_zone.md), [`Text Display`](text_display.md), [`Camera Calibration`](camera_calibration.md), [`Object Detection Model`](object_detection_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Line Counter`](line_counter.md), [`Depth Estimation`](depth_estimation.md), [`OpenAI`](open_ai.md), [`Detections Consensus`](detections_consensus.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`SAM 3`](sam3.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Seg Preview`](seg_preview.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Crop Visualization`](crop_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`Label Visualization`](label_visualization.md), [`Path Deviation`](path_deviation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Motion Detection`](motion_detection.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Dynamic Crop`](dynamic_crop.md), [`Local File Sink`](local_file_sink.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`SAM 3`](sam3.md), [`Template Matching`](template_matching.md), [`Anthropic Claude`](anthropic_claude.md), [`QR Code Generator`](qr_code_generator.md), [`YOLO-World Model`](yolo_world_model.md), [`Email Notification`](email_notification.md), [`Detections Stitch`](detections_stitch.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Distance Measurement`](distance_measurement.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Object Detection Model`](object_detection_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Size Measurement`](size_measurement.md), [`Icon Visualization`](icon_visualization.md), [`Time in Zone`](timein_zone.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Halo Visualization`](halo_visualization.md), [`Path Deviation`](path_deviation.md), [`Blur Visualization`](blur_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Email Notification`](email_notification.md), [`Florence-2 Model`](florence2_model.md), [`Webhook Sink`](webhook_sink.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`GLM-OCR`](glmocr.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`OpenAI`](open_ai.md), [`Polygon Visualization`](polygon_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemini`](google_gemini.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemini`](google_gemini.md), [`Circle Visualization`](circle_visualization.md), [`LMM`](lmm.md), [`Mask Visualization`](mask_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Image Threshold`](image_threshold.md), [`Google Gemini`](google_gemini.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Cache Get`](cache_get.md), [`Gaze Detection`](gaze_detection.md), [`Image Blur`](image_blur.md), [`Contrast Equalization`](contrast_equalization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Qwen 3.6 API`](qwen3.6_api.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Webhook Sink` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `url` (*[`string`](../kinds/string.md)*): URL of the resource to make request.
        - `query_parameters` (*Union[[`roboflow_api_key`](../kinds/roboflow_api_key.md), [`boolean`](../kinds/boolean.md), [`list_of_values`](../kinds/list_of_values.md), [`string`](../kinds/string.md), [`float`](../kinds/float.md), [`float_zero_to_one`](../kinds/float_zero_to_one.md), [`roboflow_project`](../kinds/roboflow_project.md), [`roboflow_model_id`](../kinds/roboflow_model_id.md), [`top_class`](../kinds/top_class.md), [`integer`](../kinds/integer.md)]*): Request query parameters.
        - `headers` (*Union[[`roboflow_api_key`](../kinds/roboflow_api_key.md), [`boolean`](../kinds/boolean.md), [`string`](../kinds/string.md), [`float`](../kinds/float.md), [`float_zero_to_one`](../kinds/float_zero_to_one.md), [`roboflow_project`](../kinds/roboflow_project.md), [`roboflow_model_id`](../kinds/roboflow_model_id.md), [`top_class`](../kinds/top_class.md), [`integer`](../kinds/integer.md)]*): Request headers.
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


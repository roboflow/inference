
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

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Twilio SMS/MMS Notification` in version `v2`.

    - inputs: [`Image Threshold`](image_threshold.md), [`Image Preprocessing`](image_preprocessing.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Circle Visualization`](circle_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Triangle Visualization`](triangle_visualization.md), [`Image Contours`](image_contours.md), [`Object Detection Model`](object_detection_model.md), [`Local File Sink`](local_file_sink.md), [`SORT Tracker`](sort_tracker.md), [`Qwen3-VL`](qwen3_vl.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Google Vision OCR`](google_vision_ocr.md), [`Google Gemini`](google_gemini.md), [`CSV Formatter`](csv_formatter.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`OpenAI`](open_ai.md), [`CogVLM`](cog_vlm.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Camera Focus`](camera_focus.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Line Counter`](line_counter.md), [`Mask Visualization`](mask_visualization.md), [`Color Visualization`](color_visualization.md), [`Detections Combine`](detections_combine.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Stack`](image_stack.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Path Deviation`](path_deviation.md), [`Florence-2 Model`](florence2_model.md), [`Rate Limiter`](rate_limiter.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Google Gemma`](google_gemma.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`VLM As Detector`](vlm_as_detector.md), [`Contrast Equalization`](contrast_equalization.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`GLM-OCR`](glmocr.md), [`Trace Visualization`](trace_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Seg Preview`](seg_preview.md), [`SIFT Comparison`](sift_comparison.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Detections Filter`](detections_filter.md), [`Google Gemini`](google_gemini.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`OpenAI`](open_ai.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`JSON Parser`](json_parser.md), [`Background Subtraction`](background_subtraction.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Detections Merge`](detections_merge.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Anthropic Claude`](anthropic_claude.md), [`Bounding Rectangle`](bounding_rectangle.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Path Deviation`](path_deviation.md), [`Template Matching`](template_matching.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Cosine Similarity`](cosine_similarity.md), [`Distance Measurement`](distance_measurement.md), [`Image Slicer`](image_slicer.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Google Gemma API`](google_gemma_api.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Identify Outliers`](identify_outliers.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Detection Event Log`](detection_event_log.md), [`VLM As Detector`](vlm_as_detector.md), [`Detection Offset`](detection_offset.md), [`QR Code Generator`](qr_code_generator.md), [`OpenAI`](open_ai.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Corner Visualization`](corner_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Cache Get`](cache_get.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Depth Estimation`](depth_estimation.md), [`Velocity`](velocity.md), [`Image Blur`](image_blur.md), [`Image Slicer`](image_slicer.md), [`Inner Workflow`](inner_workflow.md), [`Grid Visualization`](grid_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Halo Visualization`](halo_visualization.md), [`Overlap Analysis`](overlap_analysis.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Perspective Correction`](perspective_correction.md), [`Dominant Color`](dominant_color.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Byte Tracker`](byte_tracker.md), [`Time in Zone`](timein_zone.md), [`Byte Tracker`](byte_tracker.md), [`OpenRouter`](open_router.md), [`Detections Stitch`](detections_stitch.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Florence-2 Model`](florence2_model.md), [`Text Display`](text_display.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Camera Focus`](camera_focus.md), [`Continue If`](continue_if.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`EasyOCR`](easy_ocr.md), [`Halo Visualization`](halo_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`SIFT`](sift.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Time in Zone`](timein_zone.md), [`Webhook Sink`](webhook_sink.md), [`Property Definition`](property_definition.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Byte Tracker`](byte_tracker.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Email Notification`](email_notification.md), [`Crop Visualization`](crop_visualization.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Clip Comparison`](clip_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Moondream2`](moondream2.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Icon Visualization`](icon_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Cache Set`](cache_set.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Qwen3.5`](qwen3.5.md), [`Qwen-VL`](qwen_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`SAM 3`](sam3.md), [`Stitch Images`](stitch_images.md), [`S3 Sink`](s3_sink.md), [`SAM 3`](sam3.md), [`Detections Transformation`](detections_transformation.md), [`Line Counter`](line_counter.md), [`Gaze Detection`](gaze_detection.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Time in Zone`](timein_zone.md), [`QR Code Detection`](qr_code_detection.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenAI`](open_ai.md), [`YOLO-World Model`](yolo_world_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Delta Filter`](delta_filter.md), [`Label Visualization`](label_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Motion Detection`](motion_detection.md), [`Dimension Collapse`](dimension_collapse.md), [`LMM`](lmm.md), [`SmolVLM2`](smol_vlm2.md), [`Camera Calibration`](camera_calibration.md), [`Detections Consensus`](detections_consensus.md), [`Buffer`](buffer.md), [`SIFT Comparison`](sift_comparison.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Barcode Detection`](barcode_detection.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Identify Changes`](identify_changes.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Email Notification`](email_notification.md), [`Mask Area Measurement`](mask_area_measurement.md), [`OCR Model`](ocr_model.md), [`Expression`](expression.md), [`Overlap Filter`](overlap_filter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Size Measurement`](size_measurement.md), [`Data Aggregator`](data_aggregator.md), [`Background Color Visualization`](background_color_visualization.md), [`Slack Notification`](slack_notification.md), [`Object Detection Model`](object_detection_model.md)
    - outputs: [`Halo Visualization`](halo_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Image Threshold`](image_threshold.md), [`Image Preprocessing`](image_preprocessing.md), [`Circle Visualization`](circle_visualization.md), [`Time in Zone`](timein_zone.md), [`OpenRouter`](open_router.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Detections Stitch`](detections_stitch.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Florence-2 Model`](florence2_model.md), [`Object Detection Model`](object_detection_model.md), [`Text Display`](text_display.md), [`Local File Sink`](local_file_sink.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Google Gemini`](google_gemini.md), [`Google Vision OCR`](google_vision_ocr.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemini`](google_gemini.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI`](open_ai.md), [`CogVLM`](cog_vlm.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Halo Visualization`](halo_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Line Counter`](line_counter.md), [`Mask Visualization`](mask_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Color Visualization`](color_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Stack`](image_stack.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Path Deviation`](path_deviation.md), [`Florence-2 Model`](florence2_model.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Time in Zone`](timein_zone.md), [`Google Gemma`](google_gemma.md), [`Webhook Sink`](webhook_sink.md), [`Contrast Equalization`](contrast_equalization.md), [`GLM-OCR`](glmocr.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Email Notification`](email_notification.md), [`Seg Preview`](seg_preview.md), [`Crop Visualization`](crop_visualization.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`SIFT Comparison`](sift_comparison.md), [`Clip Comparison`](clip_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Google Gemini`](google_gemini.md), [`Moondream2`](moondream2.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Icon Visualization`](icon_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Pixel Color Count`](pixel_color_count.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Cache Set`](cache_set.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Anthropic Claude`](anthropic_claude.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Path Deviation`](path_deviation.md), [`Qwen-VL`](qwen_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Template Matching`](template_matching.md), [`SAM 3`](sam3.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`S3 Sink`](s3_sink.md), [`SAM 3`](sam3.md), [`Line Counter`](line_counter.md), [`Gaze Detection`](gaze_detection.md), [`Object Detection Model`](object_detection_model.md), [`Time in Zone`](timein_zone.md), [`Distance Measurement`](distance_measurement.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenAI`](open_ai.md), [`YOLO-World Model`](yolo_world_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Google Gemma API`](google_gemma_api.md), [`Label Visualization`](label_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Motion Detection`](motion_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`LMM`](lmm.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Camera Calibration`](camera_calibration.md), [`Detections Consensus`](detections_consensus.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`QR Code Generator`](qr_code_generator.md), [`Email Notification`](email_notification.md), [`OpenAI`](open_ai.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Corner Visualization`](corner_visualization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Cache Get`](cache_get.md), [`Polygon Visualization`](polygon_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Depth Estimation`](depth_estimation.md), [`Size Measurement`](size_measurement.md), [`Background Color Visualization`](background_color_visualization.md), [`Slack Notification`](slack_notification.md), [`Triangle Visualization`](triangle_visualization.md), [`Image Blur`](image_blur.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Twilio SMS/MMS Notification` in version `v2`  has.

???+ tip "Bindings"

    - input
    
        - `receiver_number` (*[`string`](../kinds/string.md)*): Phone number to receive the message (E.164 format, e.g., +15551234567).
        - `message_parameters` (*[`*`](../kinds/wildcard.md)*): Data to be used inside the message..
        - `media_url` (*Union[[`image`](../kinds/image.md), [`list_of_values`](../kinds/list_of_values.md), [`string`](../kinds/string.md)]*): Optional media URL(s) for MMS. Provide publicly accessible image URLs or image outputs from workflow blocks..
        - `twilio_account_sid` (*Union[[`secret`](../kinds/secret.md), [`string`](../kinds/string.md)]*): Twilio Account SID from the Twilio Console..
        - `twilio_auth_token` (*Union[[`secret`](../kinds/secret.md), [`string`](../kinds/string.md)]*): Twilio Auth Token from the Twilio Console..
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


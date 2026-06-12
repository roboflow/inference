
# Email Notification



## v2

??? "Class: `EmailNotificationBlockV2`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/sinks/email_notification/v2.py">inference.core.workflows.core_steps.sinks.email_notification.v2.EmailNotificationBlockV2</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



The **Email Notification** block allows users to send email notifications as part of a workflow.

## How This Block Works

### Email Provider Options

This block supports two email delivery methods via a dropdown selector:

1. **Roboflow Managed API Key (Default)** - No SMTP configuration needed. Emails are sent through Roboflow's proxy service:
   * **Simplified setup** - just provide subject, message, and recipient
   * **Secure** - your workflow API key is used for authentication
   * **No SMTP server required**

2. **Custom SMTP** - Use your own SMTP server:
   * Full control over email delivery
   * Requires SMTP server configuration (host, port, credentials)
   * Supports CC and BCC recipients

### Customizable Email Content

* **Subject:** Set the subject field to define the subject line of the email.

* **Message:** Use the message field to write the body content of the email. **Message can be parametrised
with data generated during workflow run. See *Dynamic Parameters* section.**

* **Recipients (To, CC, BCC)**: Define who will receive the email using `receiver_email`, 
`cc_receiver_email`, and `bcc_receiver_email` properties. You can input a single email or a list.

### Dynamic Parameters

Content of the message can be parametrised with Workflow execution outcomes. Take a look at the example
message using dynamic parameters:

```
message = "This is example notification. Predicted classes: {{ '{{' }} $parameters.predicted_classes {{ '}}' }}"
```

Message parameters are delivered by Workflows Execution Engine by setting proper data selectors in
`message_parameters` field, for example:

```
message_parameters = {
    "predicted_classes": "$steps.model.predictions"
}
```

Selecting data is not the only option - data may be processed in the block. In the example below we wish to
extract names of predicted classes. We can apply transformation **for each parameter** by setting
`message_parameters_operations`:

```
message_parameters_operations = {
    "predictions": [
        {"type": "DetectionsPropertyExtract", "property_name": "class_name"}
    ]
}
```

As a result, in the e-mail that will be sent, you can expect:

```
This is example notification. Predicted classes: ["class_a", "class_b"].
```

### Using Custom SMTP Server

To use your own SMTP server, select "Custom SMTP" from the `email_provider` dropdown and configure 
the following parameters:

* `smtp_server` - hostname of the SMTP server to use

* `sender_email` - e-mail account to be used as sender

* `sender_email_password` - password for sender e-mail account

* `smtp_port` - port of SMTP service - defaults to `465`

Block **enforces** SSL over SMTP.

Typical scenario for using custom SMTP server involves sending e-mail through Google SMTP server.
Take a look at [Google tutorial](https://support.google.com/a/answer/176600?hl=en) to configure the 
block properly. 

!!! note "GMAIL password will not work if 2-step verification is turned on"
    
    GMAIL users choosing custom SMTP server as e-mail service provider must configure 
    [application password](https://support.google.com/accounts/answer/185833) to avoid
    problems with 2-step verification protected account. Beware that **application
    password must be kept protected** - we recommend sending the password in Workflow 
    input and providing it each time by the caller, avoiding storing it in Workflow 
    definition.
    
### Cooldown

The block accepts `cooldown_seconds` (which **defaults to `5` seconds**) to prevent unintended bursts of 
notifications. Please adjust it according to your needs, setting `0` indicate no cooldown. 

During cooldown period, consecutive runs of the step will cause `throttling_status` output to be set `True`
and no notification will be sent.

!!! warning "Cooldown limitations"

    Current implementation of cooldown is limited to video processing - using this block in context of a 
    Workflow that is run behind HTTP service (Roboflow Hosted API, Dedicated Deployment or self-hosted 
    `inference` server) will have no effect for processing HTTP requests.  

### Attachments

You may specify attachment files to be sent with your e-mail. Attachments can be generated 
in runtime by dedicated blocks or from image outputs.

**Supported attachment types:**
- **CSV/Text files**: From blocks like [CSV Formatter](https://inference.roboflow.com/workflows/csv_formatter/)
- **Images**: Any image output from visualization blocks (automatically converted to JPEG)
- **Binary data**: Any bytes output from compatible blocks

To include attachments, provide the attachment filename as the key and reference the block output:

```
attachments = {
    "report.csv": "$steps.csv_formatter.csv_content",
    "detection.jpg": "$steps.bounding_box_visualization.image"
}
```

**Note:** Image attachments are automatically converted to JPEG format. If the filename doesn't 
include a `.jpg` or `.jpeg` extension, it will be added automatically.

### Async execution

Configure the `fire_and_forget` property. Set it to True if you want the email to be sent in the background, allowing the 
Workflow to proceed without waiting on e-mail to be sent. In this case you will not be able to rely on 
`error_status` output which will always be set to `False`, so we **recommend setting the `fire_and_forget=False` for
debugging purposes**.

### Disabling notifications based on runtime parameter

Sometimes it would be convenient to manually disable the e-mail notifier block. This is possible 
setting `disable_sink` flag to hold reference to Workflow input. with such setup, caller would be
able to disable the sink when needed sending agreed input parameter.

## Common Use Cases

- Use this block to [purpose based on block type]

## Connecting to Other Blocks

The outputs from this block can be connected to other blocks in your workflow.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/email_notification@v2`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `email_provider` | `str` | Choose email delivery method: use Roboflow's managed service or configure your own SMTP server.. | ❌ |
| `subject` | `str` | Subject of the message.. | ❌ |
| `receiver_email` | `Union[List[str], str]` | Destination e-mail address.. | ✅ |
| `message` | `str` | Content of the message to be send.. | ❌ |
| `message_parameters` | `Dict[str, Union[bool, float, int, str]]` | Data to be used inside the message.. | ✅ |
| `message_parameters_operations` | `Dict[str, List[Union[ClassificationPropertyExtract, ConvertDictionaryToJSON, ConvertImageToBase64, ConvertImageToJPEG, DetectionsFilter, DetectionsOffset, DetectionsPropertyExtract, DetectionsRename, DetectionsSelection, DetectionsShift, DetectionsToDictionary, Divide, ExtractDetectionProperty, ExtractFrameMetadata, ExtractImageProperty, LookupTable, Multiply, NumberRound, NumericSequenceAggregate, PickDetectionsByParentClass, RandomNumber, SequenceAggregate, SequenceApply, SequenceElementsCount, SequenceLength, SequenceMap, SortDetections, StringMatches, StringSubSequence, StringToLowerCase, StringToUpperCase, TimestampToISOFormat, ToBoolean, ToNumber, ToString]]]` | Preprocessing operations to be performed on message parameters.. | ❌ |
| `sender_email` | `str` | E-mail to be used to send the message.. | ✅ |
| `smtp_server` | `str` | Custom SMTP server to be used.. | ✅ |
| `sender_email_password` | `str` | Sender e-mail password be used when authenticating to SMTP server.. | ✅ |
| `cc_receiver_email` | `Optional[List[str], str]` | CC e-mail address.. | ✅ |
| `bcc_receiver_email` | `Optional[List[str], str]` | BCC e-mail address.. | ✅ |
| `smtp_port` | `int` | SMTP server port.. | ❌ |
| `fire_and_forget` | `bool` | Boolean flag to run the block asynchronously (True) for faster workflows or  synchronously (False) for debugging and error handling.. | ✅ |
| `disable_sink` | `bool` | Boolean flag to disable block execution.. | ✅ |
| `cooldown_seconds` | `int` | Number of seconds until a follow-up notification can be sent. . | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-alert-circle-outline:{ style="color: #f57c00" } `soft` — runtime `hosted_serverless`, `dedicated_deployment`; execution `remote`
:   Cooldown / rate-limit timer is stored in process memory. With remote step execution on stateless or multi-replica HTTP runtimes each request gets a fresh worker, so cooldown does not throttle. Cooldown only behaves as documented with local step execution inside an InferencePipeline.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Email Notification` in version `v2`.

    - inputs: [`Overlap Analysis`](overlap_analysis.md), [`Detections Transformation`](detections_transformation.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`YOLO-World Model`](yolo_world_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Anthropic Claude`](anthropic_claude.md), [`Camera Focus`](camera_focus.md), [`Track Class Lock`](track_class_lock.md), [`SmolVLM2`](smol_vlm2.md), [`Label Visualization`](label_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Qwen-VL`](qwen_vl.md), [`Text Display`](text_display.md), [`Velocity`](velocity.md), [`CSV Formatter`](csv_formatter.md), [`Gaze Detection`](gaze_detection.md), [`LMM`](lmm.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Line Counter`](line_counter.md), [`Qwen3-VL`](qwen3_vl.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemma API`](google_gemma_api.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Halo Visualization`](halo_visualization.md), [`Event Writer`](event_writer.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Property Definition`](property_definition.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Identify Outliers`](identify_outliers.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Clip Comparison`](clip_comparison.md), [`SIFT Comparison`](sift_comparison.md), [`Time in Zone`](timein_zone.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Slack Notification`](slack_notification.md), [`OpenRouter`](open_router.md), [`Detection Event Log`](detection_event_log.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Dynamic Zone`](dynamic_zone.md), [`Google Gemma`](google_gemma.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`GLM-OCR`](glmocr.md), [`Icon Visualization`](icon_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`QR Code Generator`](qr_code_generator.md), [`Path Deviation`](path_deviation.md), [`MQTT Writer`](mqtt_writer.md), [`Object Detection Model`](object_detection_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Dot Visualization`](dot_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Seg Preview`](seg_preview.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`SIFT`](sift.md), [`Local File Sink`](local_file_sink.md), [`Triangle Visualization`](triangle_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Polygon Visualization`](polygon_visualization.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Data Aggregator`](data_aggregator.md), [`Rate Limiter`](rate_limiter.md), [`Google Gemini`](google_gemini.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`LMM For Classification`](lmm_for_classification.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Stack`](image_stack.md), [`Email Notification`](email_notification.md), [`Mask Visualization`](mask_visualization.md), [`Distance Measurement`](distance_measurement.md), [`Barcode Detection`](barcode_detection.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Keypoint Visualization`](keypoint_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Overlap Filter`](overlap_filter.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Image Contours`](image_contours.md), [`Byte Tracker`](byte_tracker.md), [`SAM 3`](sam3.md), [`Motion Detection`](motion_detection.md), [`Current Time`](current_time.md), [`Corner Visualization`](corner_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Circle Visualization`](circle_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Background Color Visualization`](background_color_visualization.md), [`Line Counter`](line_counter.md), [`Template Matching`](template_matching.md), [`Morphological Transformation`](morphological_transformation.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Delta Filter`](delta_filter.md), [`OpenAI`](open_ai.md), [`Size Measurement`](size_measurement.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Trace Visualization`](trace_visualization.md), [`JSON Parser`](json_parser.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Blur`](image_blur.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`QR Code Detection`](qr_code_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Camera Focus`](camera_focus.md), [`SORT Tracker`](sort_tracker.md), [`VLM As Detector`](vlm_as_detector.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Detections Stitch`](detections_stitch.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Color Visualization`](color_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Buffer`](buffer.md), [`Cache Set`](cache_set.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Time in Zone`](timein_zone.md), [`OpenAI`](open_ai.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Detection Offset`](detection_offset.md), [`Dominant Color`](dominant_color.md), [`CogVLM`](cog_vlm.md), [`Detections Consensus`](detections_consensus.md), [`Object Detection Model`](object_detection_model.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Dynamic Crop`](dynamic_crop.md), [`Path Deviation`](path_deviation.md), [`Byte Tracker`](byte_tracker.md), [`Expression`](expression.md), [`Detections Combine`](detections_combine.md), [`Continue If`](continue_if.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Cache Get`](cache_get.md), [`SAM 3`](sam3.md), [`OpenAI`](open_ai.md), [`OCR Model`](ocr_model.md), [`Google Vision OCR`](google_vision_ocr.md), [`Pixelate Visualization`](pixelate_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Halo Visualization`](halo_visualization.md), [`Image Threshold`](image_threshold.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Stitch Images`](stitch_images.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`VLM As Classifier`](vlm_as_classifier.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemini`](google_gemini.md), [`Byte Tracker`](byte_tracker.md), [`Webhook Sink`](webhook_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Detections Merge`](detections_merge.md), [`Google Gemini`](google_gemini.md), [`Dimension Collapse`](dimension_collapse.md), [`EasyOCR`](easy_ocr.md), [`SAM 3`](sam3.md), [`Time in Zone`](timein_zone.md), [`OpenAI`](open_ai.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`VLM As Detector`](vlm_as_detector.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Identify Changes`](identify_changes.md), [`Polygon Visualization`](polygon_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Detections Filter`](detections_filter.md), [`Background Subtraction`](background_subtraction.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Email Notification`](email_notification.md), [`Image Slicer`](image_slicer.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Depth Estimation`](depth_estimation.md), [`Pixel Color Count`](pixel_color_count.md), [`Qwen3.5`](qwen3.5.md), [`Cosine Similarity`](cosine_similarity.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Camera Calibration`](camera_calibration.md), [`Inner Workflow`](inner_workflow.md), [`Grid Visualization`](grid_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Moondream2`](moondream2.md), [`S3 Sink`](s3_sink.md), [`Image Slicer`](image_slicer.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Relative Static Crop`](relative_static_crop.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Object Detection Model`](object_detection_model.md)
    - outputs: [`Template Matching`](template_matching.md), [`Morphological Transformation`](morphological_transformation.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Blur Visualization`](blur_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`OpenAI`](open_ai.md), [`YOLO-World Model`](yolo_world_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Anthropic Claude`](anthropic_claude.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Size Measurement`](size_measurement.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Trace Visualization`](trace_visualization.md), [`Label Visualization`](label_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Text Display`](text_display.md), [`Qwen-VL`](qwen_vl.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Blur`](image_blur.md), [`Gaze Detection`](gaze_detection.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`LMM`](lmm.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Line Counter`](line_counter.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Detections Stitch`](detections_stitch.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemma API`](google_gemma_api.md), [`Halo Visualization`](halo_visualization.md), [`Color Visualization`](color_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Morphological Transformation`](morphological_transformation.md), [`Event Writer`](event_writer.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Cache Set`](cache_set.md), [`Time in Zone`](timein_zone.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`OpenAI`](open_ai.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`CogVLM`](cog_vlm.md), [`Detections Consensus`](detections_consensus.md), [`Object Detection Model`](object_detection_model.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Path Deviation`](path_deviation.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`SAM 3`](sam3.md), [`Cache Get`](cache_get.md), [`OpenAI`](open_ai.md), [`Time in Zone`](timein_zone.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Slack Notification`](slack_notification.md), [`OpenRouter`](open_router.md), [`SIFT Comparison`](sift_comparison.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Dynamic Zone`](dynamic_zone.md), [`Google Gemma`](google_gemma.md), [`Halo Visualization`](halo_visualization.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`GLM-OCR`](glmocr.md), [`Image Threshold`](image_threshold.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Icon Visualization`](icon_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Google Gemini`](google_gemini.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Webhook Sink`](webhook_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`QR Code Generator`](qr_code_generator.md), [`Path Deviation`](path_deviation.md), [`MQTT Writer`](mqtt_writer.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Dot Visualization`](dot_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Seg Preview`](seg_preview.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemini`](google_gemini.md), [`Local File Sink`](local_file_sink.md), [`SAM 3`](sam3.md), [`Triangle Visualization`](triangle_visualization.md), [`Time in Zone`](timein_zone.md), [`Contrast Equalization`](contrast_equalization.md), [`Polygon Visualization`](polygon_visualization.md), [`OpenAI`](open_ai.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Google Gemini`](google_gemini.md), [`LMM For Classification`](lmm_for_classification.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Stack`](image_stack.md), [`Email Notification`](email_notification.md), [`Polygon Visualization`](polygon_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Distance Measurement`](distance_measurement.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Keypoint Visualization`](keypoint_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Email Notification`](email_notification.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`SAM 3`](sam3.md), [`Depth Estimation`](depth_estimation.md), [`Pixel Color Count`](pixel_color_count.md), [`Motion Detection`](motion_detection.md), [`Current Time`](current_time.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Corner Visualization`](corner_visualization.md), [`Moondream2`](moondream2.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`S3 Sink`](s3_sink.md), [`Circle Visualization`](circle_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Object Detection Model`](object_detection_model.md), [`Background Color Visualization`](background_color_visualization.md), [`Line Counter`](line_counter.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Email Notification` in version `v2`  has.

???+ tip "Bindings"

    - input
    
        - `receiver_email` (*Union[[`string`](../kinds/string.md), [`list_of_values`](../kinds/list_of_values.md)]*): Destination e-mail address..
        - `message_parameters` (*[`*`](../kinds/wildcard.md)*): Data to be used inside the message..
        - `sender_email` (*[`string`](../kinds/string.md)*): E-mail to be used to send the message..
        - `smtp_server` (*[`string`](../kinds/string.md)*): Custom SMTP server to be used..
        - `sender_email_password` (*Union[[`string`](../kinds/string.md), [`secret`](../kinds/secret.md)]*): Sender e-mail password be used when authenticating to SMTP server..
        - `cc_receiver_email` (*Union[[`string`](../kinds/string.md), [`list_of_values`](../kinds/list_of_values.md)]*): CC e-mail address..
        - `bcc_receiver_email` (*Union[[`string`](../kinds/string.md), [`list_of_values`](../kinds/list_of_values.md)]*): BCC e-mail address..
        - `attachments` (*Union[[`string`](../kinds/string.md), [`image`](../kinds/image.md), [`bytes`](../kinds/bytes.md)]*): Attachments.
        - `fire_and_forget` (*[`boolean`](../kinds/boolean.md)*): Boolean flag to run the block asynchronously (True) for faster workflows or  synchronously (False) for debugging and error handling..
        - `disable_sink` (*[`boolean`](../kinds/boolean.md)*): Boolean flag to disable block execution..
        - `cooldown_seconds` (*[`integer`](../kinds/integer.md)*): Number of seconds until a follow-up notification can be sent. .

    - output
    
        - `error_status` ([`boolean`](../kinds/boolean.md)): Boolean flag.
        - `throttling_status` ([`boolean`](../kinds/boolean.md)): Boolean flag.
        - `message` ([`string`](../kinds/string.md)): String value.



??? tip "Example JSON definition of step `Email Notification` in version `v2`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/email_notification@v2",
	    "email_provider": "Roboflow Managed API Key",
	    "subject": "Workflow alert",
	    "receiver_email": "receiver@gmail.com",
	    "message": "During last 5 minutes detected {{ '{{' }} $parameters.num_instances {{ '}}' }} instances",
	    "message_parameters": {
	        "predictions": "$steps.model.predictions",
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
	    "sender_email": "sender@gmail.com",
	    "smtp_server": "$inputs.smtp_server",
	    "sender_email_password": "$inputs.email_password",
	    "cc_receiver_email": "cc-receiver@gmail.com",
	    "bcc_receiver_email": "bcc-receiver@gmail.com",
	    "smtp_port": 465,
	    "attachments": {
	        "report.cvs": "$steps.csv_formatter.csv_content"
	    },
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




## v1

??? "Class: `EmailNotificationBlockV1`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/sinks/email_notification/v1.py">inference.core.workflows.core_steps.sinks.email_notification.v1.EmailNotificationBlockV1</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Send email notifications via SMTP server with customizable subject, message content with dynamic workflow data parameters, recipient lists (To, CC, BCC), file attachments, cooldown throttling, and optional async background execution for alerting, reporting, and communication workflows.

## How This Block Works

This block sends email notifications through an SMTP server, integrating workflow execution results into email content. The block:

1. Checks if the sink is disabled via `disable_sink` flag (if disabled, returns immediately without sending)
2. Validates cooldown period (if enabled, throttles notifications within `cooldown_seconds` of the last sent email, returning throttling status)
3. Formats the email message by processing dynamic parameters (replaces placeholders like `{{ '{{' }} $parameters.parameter_name {{ '}}' }}` with actual workflow data from `message_parameters`)
4. Applies optional UQL operations to transform parameter values before insertion (e.g., extract class names from detections, calculate metrics, filter data) using `message_parameters_operations`
5. Constructs email recipients lists from `receiver_email` (required), `cc_receiver_email`, and `bcc_receiver_email` (supports single email addresses or lists)
6. Processes attachments by retrieving content from referenced workflow step outputs and encoding them as email attachments
7. Establishes SSL-secured SMTP connection to the configured server (authentication using sender email and password)
8. Sends the email synchronously or asynchronously based on `fire_and_forget` setting:
   - **Synchronous mode** (`fire_and_forget=False`): Waits for email send completion, returns actual error status for debugging
   - **Asynchronous mode** (`fire_and_forget=True`): Sends email in background task, workflow continues immediately, error status always False
9. Returns status outputs indicating success, throttling, or errors

The block supports dynamic message content through parameter placeholders that are replaced with workflow data at runtime. Message parameters can be raw workflow outputs or transformed using UQL operations (e.g., extract properties, calculate counts, filter values). Attachments are sourced from other workflow blocks that produce string or binary content (e.g., CSV Formatter for reports, image outputs for screenshots). Cooldown prevents notification spam by enforcing minimum time between sends, though this only applies to video processing workflows (not HTTP request contexts).

## Requirements

**SMTP Server Configuration**: Requires access to an SMTP server with the following configuration:
- `smtp_server`: Hostname of the SMTP server (e.g., `smtp.gmail.com` for Google)
- `sender_email`: Email address to use as the sender
- `sender_email_password`: Password for the sender email account (or application-specific password for Gmail with 2FA)
- `smtp_port`: SMTP port (defaults to `465` for SSL)

**Gmail Users with 2FA**: If using Gmail with 2-step verification enabled, you must use an [application-specific password](https://support.google.com/accounts/answer/185833) instead of your regular Gmail password. Application passwords should be kept secure and provided via workflow inputs rather than stored in workflow definitions.

**Cooldown Limitations**: The cooldown mechanism (`cooldown_seconds`) only applies to video processing workflows. For HTTP request contexts (Roboflow Hosted API, Dedicated Deployment, or self-hosted servers), cooldown has no effect since each request is independent.

## Common Use Cases

- **Alert Notifications**: Send email alerts when specific conditions are detected (e.g., alert security team when unauthorized objects detected, notify operators when anomaly detected, send alerts when detection counts exceed thresholds), enabling real-time monitoring and incident response
- **Workflow Execution Reports**: Generate and email periodic or event-driven reports with workflow results (e.g., daily summary reports with detection statistics, batch processing completion notifications, performance metrics summaries), enabling automated reporting and documentation
- **Detection Summaries**: Send email summaries of detection results with aggregated statistics (e.g., email lists of detected objects, send counts and classifications, include detection confidence summaries), enabling stakeholders to stay informed about workflow outputs
- **Error and Status Notifications**: Notify administrators about workflow execution status and errors (e.g., send alerts when workflows fail, notify about processing completion, report system health issues), enabling monitoring and debugging for production deployments
- **Data Export Notifications**: Email generated data exports and reports (e.g., attach CSV reports from CSV Formatter, send exported detection data, include formatted analytics summaries), enabling automated data distribution and archival
- **Multi-Recipient Updates**: Send notifications to multiple stakeholders simultaneously using CC/BCC (e.g., notify team members about detections, send updates to multiple departments, distribute reports with CC for visibility), enabling efficient multi-party communication

## Connecting to Other Blocks

This block receives data from workflow steps and sends email notifications:

- **After detection or analysis blocks** (e.g., Object Detection, Instance Segmentation, Classification) to send alerts or summaries when objects are detected, classifications are made, or thresholds are exceeded, enabling real-time notification workflows
- **After data processing blocks** (e.g., Expression, Property Definition, Detections Filter) to include computed metrics, transformed data, or filtered results in email notifications, enabling customized reporting with processed data
- **After formatter blocks** (e.g., CSV Formatter) to attach formatted reports and exports to emails, enabling automated distribution of structured data and analytics
- **In conditional workflows** (e.g., Continue If) to send notifications only when specific conditions are met, enabling event-driven alerting and reporting
- **After aggregation blocks** (e.g., Data Aggregator) to email periodic analytics summaries and statistical reports, enabling scheduled reporting and trend analysis
- **In monitoring workflows** to send status updates, error notifications, or health check reports, enabling automated system monitoring and incident management


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/email_notification@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `subject` | `str` | Email subject line for the notification. This is the text that appears in the email header and recipient's inbox subject field. Can include static text describing the notification purpose (e.g., 'Workflow Alert', 'Detection Summary', 'Daily Report').. | ❌ |
| `sender_email` | `str` | Email address to use as the sender of the notification. This email account must have access to the configured SMTP server and the password provided in sender_email_password. For Gmail with 2FA enabled, this should be the Gmail address that has an application-specific password configured.. | ✅ |
| `receiver_email` | `Union[List[str], str]` | Primary recipient email address(es) for the notification. Required field - at least one recipient must be specified. Can be a single email address string or a list of email addresses for multiple recipients. Recipients will see their email addresses in the 'To' field of the received email.. | ✅ |
| `message` | `str` | Email body content (plain text). Supports dynamic parameters using placeholder syntax: {{ '{{' }} $parameters.parameter_name {{ '}}' }}. Placeholders are replaced with values from message_parameters at runtime. Message can be multi-line text. Example: 'Detected {{ '{{' }} $parameters.num_objects {{ '}}' }} objects. Classes: {{ '{{' }} $parameters.classes {{ '}}' }}.'. | ❌ |
| `message_parameters` | `Dict[str, Union[bool, float, int, str]]` | Dictionary mapping parameter names (used in message placeholders) to workflow data sources. Keys are parameter names referenced in message as {{ '{{' }} $parameters.key {{ '}}' }}, values are selectors to workflow step outputs or direct values. These values are substituted into message placeholders at runtime. Can optionally use message_parameters_operations to transform parameter values before substitution.. | ✅ |
| `message_parameters_operations` | `Dict[str, List[Union[ClassificationPropertyExtract, ConvertDictionaryToJSON, ConvertImageToBase64, ConvertImageToJPEG, DetectionsFilter, DetectionsOffset, DetectionsPropertyExtract, DetectionsRename, DetectionsSelection, DetectionsShift, DetectionsToDictionary, Divide, ExtractDetectionProperty, ExtractFrameMetadata, ExtractImageProperty, LookupTable, Multiply, NumberRound, NumericSequenceAggregate, PickDetectionsByParentClass, RandomNumber, SequenceAggregate, SequenceApply, SequenceElementsCount, SequenceLength, SequenceMap, SortDetections, StringMatches, StringSubSequence, StringToLowerCase, StringToUpperCase, TimestampToISOFormat, ToBoolean, ToNumber, ToString]]]` | Optional dictionary mapping parameter names (from message_parameters) to UQL operation chains that transform parameter values before inserting them into the message. Operations are applied in sequence (e.g., extract class names from detections, calculate counts, filter values). Keys must match parameter names in message_parameters. Leave empty or omit parameters that don't need transformation.. | ❌ |
| `cc_receiver_email` | `Optional[List[str], str]` | Optional CC (Carbon Copy) recipient email address(es). Can be a single email address string or a list of email addresses. CC recipients receive a copy of the email and can see each other's addresses. Use for recipients who should be informed but don't need to take action.. | ✅ |
| `bcc_receiver_email` | `Optional[List[str], str]` | Optional BCC (Blind Carbon Copy) recipient email address(es). Can be a single email address string or a list of email addresses. BCC recipients receive a copy of the email but their addresses are hidden from other recipients. Use for recipients who should receive the notification privately.. | ✅ |
| `smtp_server` | `str` | SMTP server hostname to use for sending emails. Common examples: 'smtp.gmail.com' for Gmail, 'smtp.outlook.com' for Outlook, or your organization's SMTP server hostname. The block enforces SSL/TLS encryption for SMTP connections. Ensure the server supports SSL on the specified port.. | ✅ |
| `sender_email_password` | `str` | Password for the sender email account to authenticate with the SMTP server. For Gmail with 2-step verification enabled, use an application-specific password instead of the regular Gmail password. This field is marked as private for security. Recommended to provide via workflow inputs rather than storing in workflow definitions. For Roboflow-hosted services, can use SECRET_KIND selectors for secure credential management.. | ✅ |
| `smtp_port` | `int` | SMTP server port number. Defaults to 465 (standard SSL port for SMTP). Common alternatives: 587 for TLS (not supported - this block enforces SSL), 25 for unencrypted (not recommended). Ensure the port supports SSL encryption as required by this block.. | ❌ |
| `fire_and_forget` | `bool` | Execution mode: True for asynchronous background sending (workflow continues immediately, error_status always False, faster execution), False for synchronous sending (waits for email completion, returns actual error status for debugging). Set to False during development and debugging to catch email sending errors. Set to True in production for faster workflow execution when email delivery timing is not critical.. | ✅ |
| `disable_sink` | `bool` | Flag to disable email sending at runtime. When True, the block skips sending email and returns a disabled message. Useful for conditional notification control via workflow inputs (e.g., allow callers to disable notifications for testing, enable/disable based on configuration). Set via workflow inputs for runtime control.. | ✅ |
| `cooldown_seconds` | `int` | Minimum seconds between consecutive email notifications to prevent notification spam. Defaults to 5 seconds. Set to 0 to disable cooldown (no throttling). During cooldown period, the block returns throttling_status=True and skips sending. Note: Cooldown only applies to video processing workflows, not HTTP request contexts (Roboflow Hosted API, Dedicated Deployment, or self-hosted servers where each request is independent).. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-alert-circle-outline:{ style="color: #f57c00" } `soft` — runtime `hosted_serverless`, `dedicated_deployment`; execution `remote`
:   Cooldown / rate-limit timer is stored in process memory. With remote step execution on stateless or multi-replica HTTP runtimes each request gets a fresh worker, so cooldown does not throttle. Cooldown only behaves as documented with local step execution inside an InferencePipeline.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Email Notification` in version `v1`.

    - inputs: [`Overlap Analysis`](overlap_analysis.md), [`Template Matching`](template_matching.md), [`Morphological Transformation`](morphological_transformation.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Blur Visualization`](blur_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Delta Filter`](delta_filter.md), [`OpenAI`](open_ai.md), [`YOLO-World Model`](yolo_world_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Anthropic Claude`](anthropic_claude.md), [`Camera Focus`](camera_focus.md), [`Track Class Lock`](track_class_lock.md), [`Size Measurement`](size_measurement.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Trace Visualization`](trace_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`SmolVLM2`](smol_vlm2.md), [`JSON Parser`](json_parser.md), [`Label Visualization`](label_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Florence-2 Model`](florence2_model.md), [`Qwen-VL`](qwen_vl.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Text Display`](text_display.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Blur`](image_blur.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Velocity`](velocity.md), [`CSV Formatter`](csv_formatter.md), [`Gaze Detection`](gaze_detection.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`LMM`](lmm.md), [`QR Code Detection`](qr_code_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Camera Focus`](camera_focus.md), [`Line Counter`](line_counter.md), [`SORT Tracker`](sort_tracker.md), [`VLM As Detector`](vlm_as_detector.md), [`Qwen3-VL`](qwen3_vl.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Clip Comparison`](clip_comparison.md), [`Detections Stitch`](detections_stitch.md), [`Google Gemma API`](google_gemma_api.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Halo Visualization`](halo_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Color Visualization`](color_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Event Writer`](event_writer.md), [`Buffer`](buffer.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Property Definition`](property_definition.md), [`Cache Set`](cache_set.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Time in Zone`](timein_zone.md), [`OpenAI`](open_ai.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Identify Outliers`](identify_outliers.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Detection Offset`](detection_offset.md), [`Dominant Color`](dominant_color.md), [`CogVLM`](cog_vlm.md), [`Detections Consensus`](detections_consensus.md), [`Object Detection Model`](object_detection_model.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Path Deviation`](path_deviation.md), [`Byte Tracker`](byte_tracker.md), [`Expression`](expression.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Detections Combine`](detections_combine.md), [`Continue If`](continue_if.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Clip Comparison`](clip_comparison.md), [`Cache Get`](cache_get.md), [`SAM 3`](sam3.md), [`OpenAI`](open_ai.md), [`SIFT Comparison`](sift_comparison.md), [`Time in Zone`](timein_zone.md), [`OCR Model`](ocr_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Slack Notification`](slack_notification.md), [`OpenRouter`](open_router.md), [`Detection Event Log`](detection_event_log.md), [`Google Vision OCR`](google_vision_ocr.md), [`Pixelate Visualization`](pixelate_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Google Gemma`](google_gemma.md), [`Dynamic Zone`](dynamic_zone.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Halo Visualization`](halo_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`GLM-OCR`](glmocr.md), [`Image Threshold`](image_threshold.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Stitch Images`](stitch_images.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Icon Visualization`](icon_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Google Gemini`](google_gemini.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Byte Tracker`](byte_tracker.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Webhook Sink`](webhook_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`QR Code Generator`](qr_code_generator.md), [`Path Deviation`](path_deviation.md), [`MQTT Writer`](mqtt_writer.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Dot Visualization`](dot_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Seg Preview`](seg_preview.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Detections Merge`](detections_merge.md), [`SIFT`](sift.md), [`Google Gemini`](google_gemini.md), [`EasyOCR`](easy_ocr.md), [`Dimension Collapse`](dimension_collapse.md), [`Local File Sink`](local_file_sink.md), [`SAM 3`](sam3.md), [`Triangle Visualization`](triangle_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Time in Zone`](timein_zone.md), [`Polygon Visualization`](polygon_visualization.md), [`OpenAI`](open_ai.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Data Aggregator`](data_aggregator.md), [`Rate Limiter`](rate_limiter.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Google Gemini`](google_gemini.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`LMM For Classification`](lmm_for_classification.md), [`VLM As Detector`](vlm_as_detector.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Identify Changes`](identify_changes.md), [`Image Stack`](image_stack.md), [`Email Notification`](email_notification.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Mask Visualization`](mask_visualization.md), [`Detections Filter`](detections_filter.md), [`Distance Measurement`](distance_measurement.md), [`Barcode Detection`](barcode_detection.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Keypoint Visualization`](keypoint_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Overlap Filter`](overlap_filter.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Email Notification`](email_notification.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Image Slicer`](image_slicer.md), [`Image Contours`](image_contours.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Image Preprocessing`](image_preprocessing.md), [`SAM 3`](sam3.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Depth Estimation`](depth_estimation.md), [`Pixel Color Count`](pixel_color_count.md), [`Motion Detection`](motion_detection.md), [`Current Time`](current_time.md), [`Qwen3.5`](qwen3.5.md), [`Cosine Similarity`](cosine_similarity.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Corner Visualization`](corner_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Inner Workflow`](inner_workflow.md), [`Grid Visualization`](grid_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Moondream2`](moondream2.md), [`S3 Sink`](s3_sink.md), [`Circle Visualization`](circle_visualization.md), [`Image Slicer`](image_slicer.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Relative Static Crop`](relative_static_crop.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Object Detection Model`](object_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Line Counter`](line_counter.md)
    - outputs: [`Template Matching`](template_matching.md), [`Morphological Transformation`](morphological_transformation.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Blur Visualization`](blur_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`OpenAI`](open_ai.md), [`YOLO-World Model`](yolo_world_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Anthropic Claude`](anthropic_claude.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Size Measurement`](size_measurement.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Trace Visualization`](trace_visualization.md), [`Label Visualization`](label_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Text Display`](text_display.md), [`Qwen-VL`](qwen_vl.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Blur`](image_blur.md), [`Gaze Detection`](gaze_detection.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`LMM`](lmm.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Line Counter`](line_counter.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Detections Stitch`](detections_stitch.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemma API`](google_gemma_api.md), [`Halo Visualization`](halo_visualization.md), [`Color Visualization`](color_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Morphological Transformation`](morphological_transformation.md), [`Event Writer`](event_writer.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Cache Set`](cache_set.md), [`Time in Zone`](timein_zone.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`OpenAI`](open_ai.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`CogVLM`](cog_vlm.md), [`Detections Consensus`](detections_consensus.md), [`Object Detection Model`](object_detection_model.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Path Deviation`](path_deviation.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`SAM 3`](sam3.md), [`Cache Get`](cache_get.md), [`OpenAI`](open_ai.md), [`Time in Zone`](timein_zone.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Slack Notification`](slack_notification.md), [`OpenRouter`](open_router.md), [`SIFT Comparison`](sift_comparison.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Dynamic Zone`](dynamic_zone.md), [`Google Gemma`](google_gemma.md), [`Halo Visualization`](halo_visualization.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`GLM-OCR`](glmocr.md), [`Image Threshold`](image_threshold.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Icon Visualization`](icon_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Google Gemini`](google_gemini.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Webhook Sink`](webhook_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`QR Code Generator`](qr_code_generator.md), [`Path Deviation`](path_deviation.md), [`MQTT Writer`](mqtt_writer.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Dot Visualization`](dot_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Seg Preview`](seg_preview.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemini`](google_gemini.md), [`Local File Sink`](local_file_sink.md), [`SAM 3`](sam3.md), [`Triangle Visualization`](triangle_visualization.md), [`Time in Zone`](timein_zone.md), [`Contrast Equalization`](contrast_equalization.md), [`Polygon Visualization`](polygon_visualization.md), [`OpenAI`](open_ai.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Google Gemini`](google_gemini.md), [`LMM For Classification`](lmm_for_classification.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Stack`](image_stack.md), [`Email Notification`](email_notification.md), [`Polygon Visualization`](polygon_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Distance Measurement`](distance_measurement.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Keypoint Visualization`](keypoint_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Email Notification`](email_notification.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`SAM 3`](sam3.md), [`Depth Estimation`](depth_estimation.md), [`Pixel Color Count`](pixel_color_count.md), [`Motion Detection`](motion_detection.md), [`Current Time`](current_time.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Corner Visualization`](corner_visualization.md), [`Moondream2`](moondream2.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`S3 Sink`](s3_sink.md), [`Circle Visualization`](circle_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Object Detection Model`](object_detection_model.md), [`Background Color Visualization`](background_color_visualization.md), [`Line Counter`](line_counter.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Email Notification` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `sender_email` (*[`string`](../kinds/string.md)*): Email address to use as the sender of the notification. This email account must have access to the configured SMTP server and the password provided in sender_email_password. For Gmail with 2FA enabled, this should be the Gmail address that has an application-specific password configured..
        - `receiver_email` (*Union[[`string`](../kinds/string.md), [`list_of_values`](../kinds/list_of_values.md)]*): Primary recipient email address(es) for the notification. Required field - at least one recipient must be specified. Can be a single email address string or a list of email addresses for multiple recipients. Recipients will see their email addresses in the 'To' field of the received email..
        - `message_parameters` (*[`*`](../kinds/wildcard.md)*): Dictionary mapping parameter names (used in message placeholders) to workflow data sources. Keys are parameter names referenced in message as {{ '{{' }} $parameters.key {{ '}}' }}, values are selectors to workflow step outputs or direct values. These values are substituted into message placeholders at runtime. Can optionally use message_parameters_operations to transform parameter values before substitution..
        - `cc_receiver_email` (*Union[[`string`](../kinds/string.md), [`list_of_values`](../kinds/list_of_values.md)]*): Optional CC (Carbon Copy) recipient email address(es). Can be a single email address string or a list of email addresses. CC recipients receive a copy of the email and can see each other's addresses. Use for recipients who should be informed but don't need to take action..
        - `bcc_receiver_email` (*Union[[`string`](../kinds/string.md), [`list_of_values`](../kinds/list_of_values.md)]*): Optional BCC (Blind Carbon Copy) recipient email address(es). Can be a single email address string or a list of email addresses. BCC recipients receive a copy of the email but their addresses are hidden from other recipients. Use for recipients who should receive the notification privately..
        - `attachments` (*Union[[`string`](../kinds/string.md), [`bytes`](../kinds/bytes.md)]*): Optional dictionary mapping attachment filenames to workflow step outputs that provide file content. Keys are the attachment filenames (e.g., 'report.csv', 'summary.pdf'), values are selectors to blocks that output string or binary content (e.g., CSV Formatter outputs, image data, generated reports). Attachments are encoded and attached to the email. Leave empty if no attachments are needed..
        - `smtp_server` (*[`string`](../kinds/string.md)*): SMTP server hostname to use for sending emails. Common examples: 'smtp.gmail.com' for Gmail, 'smtp.outlook.com' for Outlook, or your organization's SMTP server hostname. The block enforces SSL/TLS encryption for SMTP connections. Ensure the server supports SSL on the specified port..
        - `sender_email_password` (*Union[[`string`](../kinds/string.md), [`secret`](../kinds/secret.md)]*): Password for the sender email account to authenticate with the SMTP server. For Gmail with 2-step verification enabled, use an application-specific password instead of the regular Gmail password. This field is marked as private for security. Recommended to provide via workflow inputs rather than storing in workflow definitions. For Roboflow-hosted services, can use SECRET_KIND selectors for secure credential management..
        - `fire_and_forget` (*[`boolean`](../kinds/boolean.md)*): Execution mode: True for asynchronous background sending (workflow continues immediately, error_status always False, faster execution), False for synchronous sending (waits for email completion, returns actual error status for debugging). Set to False during development and debugging to catch email sending errors. Set to True in production for faster workflow execution when email delivery timing is not critical..
        - `disable_sink` (*[`boolean`](../kinds/boolean.md)*): Flag to disable email sending at runtime. When True, the block skips sending email and returns a disabled message. Useful for conditional notification control via workflow inputs (e.g., allow callers to disable notifications for testing, enable/disable based on configuration). Set via workflow inputs for runtime control..
        - `cooldown_seconds` (*[`integer`](../kinds/integer.md)*): Minimum seconds between consecutive email notifications to prevent notification spam. Defaults to 5 seconds. Set to 0 to disable cooldown (no throttling). During cooldown period, the block returns throttling_status=True and skips sending. Note: Cooldown only applies to video processing workflows, not HTTP request contexts (Roboflow Hosted API, Dedicated Deployment, or self-hosted servers where each request is independent)..

    - output
    
        - `error_status` ([`boolean`](../kinds/boolean.md)): Boolean flag.
        - `throttling_status` ([`boolean`](../kinds/boolean.md)): Boolean flag.
        - `message` ([`string`](../kinds/string.md)): String value.



??? tip "Example JSON definition of step `Email Notification` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/email_notification@v1",
	    "subject": "Workflow alert",
	    "sender_email": "sender@gmail.com",
	    "receiver_email": "receiver@gmail.com",
	    "message": "During last 5 minutes detected {{ '{{' }} $parameters.num_instances {{ '}}' }} instances",
	    "message_parameters": {
	        "predictions": "$steps.model.predictions",
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
	    "cc_receiver_email": "cc-receiver@gmail.com",
	    "bcc_receiver_email": "bcc-receiver@gmail.com",
	    "attachments": {
	        "report.cvs": "$steps.csv_formatter.csv_content"
	    },
	    "smtp_server": "$inputs.smtp_server",
	    "sender_email_password": "$inputs.email_password",
	    "smtp_port": 465,
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


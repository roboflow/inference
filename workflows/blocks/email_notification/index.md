
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

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Email Notification` in version `v2`.

    - inputs: [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`OCR Model`](ocr_model.md), [`Gaze Detection`](gaze_detection.md), [`Image Slicer`](image_slicer.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Distance Measurement`](distance_measurement.md), [`Color Visualization`](color_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Data Aggregator`](data_aggregator.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Byte Tracker`](byte_tracker.md), [`Detections Consensus`](detections_consensus.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Barcode Detection`](barcode_detection.md), [`Cache Set`](cache_set.md), [`Webhook Sink`](webhook_sink.md), [`Continue If`](continue_if.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Trace Visualization`](trace_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Camera Focus`](camera_focus.md), [`Object Detection Model`](object_detection_model.md), [`OpenAI`](open_ai.md), [`Buffer`](buffer.md), [`SAM 3`](sam3.md), [`Size Measurement`](size_measurement.md), [`Image Threshold`](image_threshold.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Florence-2 Model`](florence2_model.md), [`Halo Visualization`](halo_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Path Deviation`](path_deviation.md), [`GLM-OCR`](glmocr.md), [`Dot Visualization`](dot_visualization.md), [`S3 Sink`](s3_sink.md), [`Path Deviation`](path_deviation.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Seg Preview`](seg_preview.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Dynamic Zone`](dynamic_zone.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Line Counter`](line_counter.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Text Display`](text_display.md), [`CSV Formatter`](csv_formatter.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Detections Merge`](detections_merge.md), [`Perspective Correction`](perspective_correction.md), [`Anthropic Claude`](anthropic_claude.md), [`Line Counter`](line_counter.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Velocity`](velocity.md), [`Depth Estimation`](depth_estimation.md), [`Overlap Filter`](overlap_filter.md), [`Rate Limiter`](rate_limiter.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Polygon Visualization`](polygon_visualization.md), [`SmolVLM2`](smol_vlm2.md), [`SIFT`](sift.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemini`](google_gemini.md), [`Label Visualization`](label_visualization.md), [`Expression`](expression.md), [`Grid Visualization`](grid_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Contrast Equalization`](contrast_equalization.md), [`Triangle Visualization`](triangle_visualization.md), [`Property Definition`](property_definition.md), [`Halo Visualization`](halo_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`Dominant Color`](dominant_color.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`Slack Notification`](slack_notification.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Detections Stitch`](detections_stitch.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Object Detection Model`](object_detection_model.md), [`Email Notification`](email_notification.md), [`Google Gemma API`](google_gemma_api.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Google Vision OCR`](google_vision_ocr.md), [`Identify Outliers`](identify_outliers.md), [`Google Gemini`](google_gemini.md), [`Image Preprocessing`](image_preprocessing.md), [`EasyOCR`](easy_ocr.md), [`Detections Combine`](detections_combine.md), [`Object Detection Model`](object_detection_model.md), [`Cosine Similarity`](cosine_similarity.md), [`OpenAI`](open_ai.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Byte Tracker`](byte_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Time in Zone`](timein_zone.md), [`Qwen3-VL`](qwen3_vl.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`YOLO-World Model`](yolo_world_model.md), [`Inner Workflow`](inner_workflow.md), [`Detection Offset`](detection_offset.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Template Matching`](template_matching.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Stitch Images`](stitch_images.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`CogVLM`](cog_vlm.md), [`Crop Visualization`](crop_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Camera Calibration`](camera_calibration.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Time in Zone`](timein_zone.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`SAM 3`](sam3.md), [`QR Code Detection`](qr_code_detection.md), [`Local File Sink`](local_file_sink.md), [`Icon Visualization`](icon_visualization.md), [`Detections Filter`](detections_filter.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Image Contours`](image_contours.md), [`JSON Parser`](json_parser.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Time in Zone`](timein_zone.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`Anthropic Claude`](anthropic_claude.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Detector`](vlm_as_detector.md), [`LMM`](lmm.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Pixel Color Count`](pixel_color_count.md), [`Identify Changes`](identify_changes.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Image Blur`](image_blur.md), [`Image Slicer`](image_slicer.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`SAM 3`](sam3.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Corner Visualization`](corner_visualization.md), [`Delta Filter`](delta_filter.md), [`Dynamic Crop`](dynamic_crop.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Moondream2`](moondream2.md), [`Cache Get`](cache_get.md), [`QR Code Generator`](qr_code_generator.md), [`Camera Focus`](camera_focus.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Background Color Visualization`](background_color_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`SIFT Comparison`](sift_comparison.md)
    - outputs: [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Gaze Detection`](gaze_detection.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Distance Measurement`](distance_measurement.md), [`Color Visualization`](color_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Detections Consensus`](detections_consensus.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Cache Set`](cache_set.md), [`Webhook Sink`](webhook_sink.md), [`Trace Visualization`](trace_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`OpenAI`](open_ai.md), [`SAM 3`](sam3.md), [`Size Measurement`](size_measurement.md), [`Image Threshold`](image_threshold.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Halo Visualization`](halo_visualization.md), [`Path Deviation`](path_deviation.md), [`GLM-OCR`](glmocr.md), [`Dot Visualization`](dot_visualization.md), [`S3 Sink`](s3_sink.md), [`Path Deviation`](path_deviation.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Seg Preview`](seg_preview.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Dynamic Zone`](dynamic_zone.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Line Counter`](line_counter.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`Text Display`](text_display.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Perspective Correction`](perspective_correction.md), [`Anthropic Claude`](anthropic_claude.md), [`Line Counter`](line_counter.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Depth Estimation`](depth_estimation.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Google Gemini`](google_gemini.md), [`Label Visualization`](label_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Triangle Visualization`](triangle_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`Slack Notification`](slack_notification.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Detections Stitch`](detections_stitch.md), [`Object Detection Model`](object_detection_model.md), [`Email Notification`](email_notification.md), [`Google Gemma API`](google_gemma_api.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Google Vision OCR`](google_vision_ocr.md), [`Google Gemini`](google_gemini.md), [`Image Preprocessing`](image_preprocessing.md), [`Object Detection Model`](object_detection_model.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Time in Zone`](timein_zone.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`YOLO-World Model`](yolo_world_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Template Matching`](template_matching.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`CogVLM`](cog_vlm.md), [`Crop Visualization`](crop_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Florence-2 Model`](florence2_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Time in Zone`](timein_zone.md), [`SAM 3`](sam3.md), [`Icon Visualization`](icon_visualization.md), [`Local File Sink`](local_file_sink.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Time in Zone`](timein_zone.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Clip Comparison`](clip_comparison.md), [`LMM`](lmm.md), [`Pixel Color Count`](pixel_color_count.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Image Blur`](image_blur.md), [`SAM 3`](sam3.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Corner Visualization`](corner_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Moondream2`](moondream2.md), [`QR Code Generator`](qr_code_generator.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Background Color Visualization`](background_color_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Cache Get`](cache_get.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Email Notification` in version `v2`  has.

???+ tip "Bindings"

    - input
    
        - `receiver_email` (*Union[[`string`](../kinds/string.md), [`list_of_values`](../kinds/list_of_values.md)]*): Destination e-mail address..
        - `message_parameters` (*[`*`](../kinds/wildcard.md)*): Data to be used inside the message..
        - `sender_email` (*[`string`](../kinds/string.md)*): E-mail to be used to send the message..
        - `smtp_server` (*[`string`](../kinds/string.md)*): Custom SMTP server to be used..
        - `sender_email_password` (*Union[[`secret`](../kinds/secret.md), [`string`](../kinds/string.md)]*): Sender e-mail password be used when authenticating to SMTP server..
        - `cc_receiver_email` (*Union[[`string`](../kinds/string.md), [`list_of_values`](../kinds/list_of_values.md)]*): CC e-mail address..
        - `bcc_receiver_email` (*Union[[`string`](../kinds/string.md), [`list_of_values`](../kinds/list_of_values.md)]*): BCC e-mail address..
        - `attachments` (*Union[[`bytes`](../kinds/bytes.md), [`string`](../kinds/string.md), [`image`](../kinds/image.md)]*): Attachments.
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

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Email Notification` in version `v1`.

    - inputs: [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`OCR Model`](ocr_model.md), [`Gaze Detection`](gaze_detection.md), [`Image Slicer`](image_slicer.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Distance Measurement`](distance_measurement.md), [`Color Visualization`](color_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Data Aggregator`](data_aggregator.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Byte Tracker`](byte_tracker.md), [`Detections Consensus`](detections_consensus.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Barcode Detection`](barcode_detection.md), [`Cache Set`](cache_set.md), [`Webhook Sink`](webhook_sink.md), [`Continue If`](continue_if.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Trace Visualization`](trace_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Camera Focus`](camera_focus.md), [`Object Detection Model`](object_detection_model.md), [`OpenAI`](open_ai.md), [`Buffer`](buffer.md), [`SAM 3`](sam3.md), [`Size Measurement`](size_measurement.md), [`Image Threshold`](image_threshold.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Florence-2 Model`](florence2_model.md), [`Halo Visualization`](halo_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Path Deviation`](path_deviation.md), [`GLM-OCR`](glmocr.md), [`Dot Visualization`](dot_visualization.md), [`S3 Sink`](s3_sink.md), [`Path Deviation`](path_deviation.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Seg Preview`](seg_preview.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Dynamic Zone`](dynamic_zone.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Line Counter`](line_counter.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Text Display`](text_display.md), [`CSV Formatter`](csv_formatter.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Detections Merge`](detections_merge.md), [`Perspective Correction`](perspective_correction.md), [`Anthropic Claude`](anthropic_claude.md), [`Line Counter`](line_counter.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Velocity`](velocity.md), [`Depth Estimation`](depth_estimation.md), [`Overlap Filter`](overlap_filter.md), [`Rate Limiter`](rate_limiter.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Polygon Visualization`](polygon_visualization.md), [`SmolVLM2`](smol_vlm2.md), [`SIFT`](sift.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemini`](google_gemini.md), [`Label Visualization`](label_visualization.md), [`Expression`](expression.md), [`Grid Visualization`](grid_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Contrast Equalization`](contrast_equalization.md), [`Triangle Visualization`](triangle_visualization.md), [`Property Definition`](property_definition.md), [`Halo Visualization`](halo_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`Dominant Color`](dominant_color.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`Slack Notification`](slack_notification.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Detections Stitch`](detections_stitch.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Object Detection Model`](object_detection_model.md), [`Email Notification`](email_notification.md), [`Google Gemma API`](google_gemma_api.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Google Vision OCR`](google_vision_ocr.md), [`Identify Outliers`](identify_outliers.md), [`Google Gemini`](google_gemini.md), [`Image Preprocessing`](image_preprocessing.md), [`EasyOCR`](easy_ocr.md), [`Detections Combine`](detections_combine.md), [`Object Detection Model`](object_detection_model.md), [`Cosine Similarity`](cosine_similarity.md), [`OpenAI`](open_ai.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Byte Tracker`](byte_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Time in Zone`](timein_zone.md), [`Qwen3-VL`](qwen3_vl.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`YOLO-World Model`](yolo_world_model.md), [`Inner Workflow`](inner_workflow.md), [`Detection Offset`](detection_offset.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Template Matching`](template_matching.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Stitch Images`](stitch_images.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`CogVLM`](cog_vlm.md), [`Crop Visualization`](crop_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Camera Calibration`](camera_calibration.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Time in Zone`](timein_zone.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`SAM 3`](sam3.md), [`QR Code Detection`](qr_code_detection.md), [`Local File Sink`](local_file_sink.md), [`Icon Visualization`](icon_visualization.md), [`Detections Filter`](detections_filter.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Image Contours`](image_contours.md), [`JSON Parser`](json_parser.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Time in Zone`](timein_zone.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`Anthropic Claude`](anthropic_claude.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Detector`](vlm_as_detector.md), [`LMM`](lmm.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Pixel Color Count`](pixel_color_count.md), [`Identify Changes`](identify_changes.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Image Blur`](image_blur.md), [`Image Slicer`](image_slicer.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`SAM 3`](sam3.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Corner Visualization`](corner_visualization.md), [`Delta Filter`](delta_filter.md), [`Dynamic Crop`](dynamic_crop.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Moondream2`](moondream2.md), [`Cache Get`](cache_get.md), [`QR Code Generator`](qr_code_generator.md), [`Camera Focus`](camera_focus.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Background Color Visualization`](background_color_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`SIFT Comparison`](sift_comparison.md)
    - outputs: [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Gaze Detection`](gaze_detection.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Distance Measurement`](distance_measurement.md), [`Color Visualization`](color_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Detections Consensus`](detections_consensus.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Cache Set`](cache_set.md), [`Webhook Sink`](webhook_sink.md), [`Trace Visualization`](trace_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`OpenAI`](open_ai.md), [`SAM 3`](sam3.md), [`Size Measurement`](size_measurement.md), [`Image Threshold`](image_threshold.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Halo Visualization`](halo_visualization.md), [`Path Deviation`](path_deviation.md), [`GLM-OCR`](glmocr.md), [`Dot Visualization`](dot_visualization.md), [`S3 Sink`](s3_sink.md), [`Path Deviation`](path_deviation.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Seg Preview`](seg_preview.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Dynamic Zone`](dynamic_zone.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Line Counter`](line_counter.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`Text Display`](text_display.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Perspective Correction`](perspective_correction.md), [`Anthropic Claude`](anthropic_claude.md), [`Line Counter`](line_counter.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Depth Estimation`](depth_estimation.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Google Gemini`](google_gemini.md), [`Label Visualization`](label_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Triangle Visualization`](triangle_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`Slack Notification`](slack_notification.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Detections Stitch`](detections_stitch.md), [`Object Detection Model`](object_detection_model.md), [`Email Notification`](email_notification.md), [`Google Gemma API`](google_gemma_api.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Google Vision OCR`](google_vision_ocr.md), [`Google Gemini`](google_gemini.md), [`Image Preprocessing`](image_preprocessing.md), [`Object Detection Model`](object_detection_model.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Time in Zone`](timein_zone.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`YOLO-World Model`](yolo_world_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Template Matching`](template_matching.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`CogVLM`](cog_vlm.md), [`Crop Visualization`](crop_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Florence-2 Model`](florence2_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Time in Zone`](timein_zone.md), [`SAM 3`](sam3.md), [`Icon Visualization`](icon_visualization.md), [`Local File Sink`](local_file_sink.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Time in Zone`](timein_zone.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Clip Comparison`](clip_comparison.md), [`LMM`](lmm.md), [`Pixel Color Count`](pixel_color_count.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Image Blur`](image_blur.md), [`SAM 3`](sam3.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Corner Visualization`](corner_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Moondream2`](moondream2.md), [`QR Code Generator`](qr_code_generator.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Background Color Visualization`](background_color_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Cache Get`](cache_get.md)

    
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
        - `attachments` (*Union[[`bytes`](../kinds/bytes.md), [`string`](../kinds/string.md)]*): Optional dictionary mapping attachment filenames to workflow step outputs that provide file content. Keys are the attachment filenames (e.g., 'report.csv', 'summary.pdf'), values are selectors to blocks that output string or binary content (e.g., CSV Formatter outputs, image data, generated reports). Attachments are encoded and attached to the email. Leave empty if no attachments are needed..
        - `smtp_server` (*[`string`](../kinds/string.md)*): SMTP server hostname to use for sending emails. Common examples: 'smtp.gmail.com' for Gmail, 'smtp.outlook.com' for Outlook, or your organization's SMTP server hostname. The block enforces SSL/TLS encryption for SMTP connections. Ensure the server supports SSL on the specified port..
        - `sender_email_password` (*Union[[`secret`](../kinds/secret.md), [`string`](../kinds/string.md)]*): Password for the sender email account to authenticate with the SMTP server. For Gmail with 2-step verification enabled, use an application-specific password instead of the regular Gmail password. This field is marked as private for security. Recommended to provide via workflow inputs rather than storing in workflow definitions. For Roboflow-hosted services, can use SECRET_KIND selectors for secure credential management..
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


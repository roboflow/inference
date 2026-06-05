
# S3 Sink



??? "Class: `S3SinkBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/sinks/s3/v1.py">inference.core.workflows.core_steps.sinks.s3.v1.S3SinkBlockV1</a>
    



Save workflow data directly to an AWS S3 bucket, supporting CSV, JSON, and text file formats with configurable output modes for aggregating multiple entries into single objects or saving each entry as a separate S3 object.

## How This Block Works

This block uploads string content from workflow steps to S3 objects. The block:

1. Takes string content (from formatters, predictions, or other string-producing blocks) and S3 configuration as input
2. Connects to AWS S3 using the provided credentials (or the default AWS credential chain if none are supplied)
3. Selects the appropriate upload strategy based on `output_mode`:
   - **Separate Files Mode**: Creates a new S3 object for each input, generating unique keys with timestamps
   - **Append Log Mode**: Buffers content in memory, uploading a complete object when `max_entries_per_file` is reached or when the block is destroyed
4. For **separate files mode**: Generates a unique S3 key from the prefix, file name prefix, file type, and a timestamp, then uploads the content directly
5. For **append log mode**:
   - Buffers content entries in memory under a single S3 key
   - Applies format-specific handling for appending:
     - **CSV**: Removes the header row from subsequent appends (CSV content must include headers on first write)
     - **JSON**: Converts to JSONL (JSON Lines) format, parsing and re-serializing each JSON document to fit on a single line
     - **TXT**: Appends content directly with newlines
   - Tracks entry count and uploads the full buffer as a complete S3 object when `max_entries_per_file` is reached, then starts a fresh buffer with a new key
   - Uploads any remaining buffered data when the block is destroyed
6. Returns error status and messages indicating save success or failure

The block supports two storage strategies: separate files mode creates individual timestamped S3 objects per input (useful for organizing outputs by execution), while append log mode accumulates entries in memory and writes them as complete S3 objects on rotation (useful for time-series logging with controlled upload frequency). S3 key names include timestamps (format: `YYYY_MM_DD_HH_MM_SS_microseconds`) for unique keys and chronological ordering.

## AWS Credentials

Credentials can be supplied in two ways:
1. **Workflow inputs** — declare `aws_access_key_id` and `aws_secret_access_key` as workflow inputs of kind `parameter` and connect them to the corresponding fields. This keeps credentials out of the workflow definition and allows them to be supplied at runtime.
2. **Secrets provider block** — connect the credential fields to the output of an `Environment Secrets Store` block, which reads values from server-side environment variables without embedding them in the workflow. Note: this is only available on self-hosted `inference` servers and cannot be used on the Roboflow hosted platform.

## S3 Key Structure

The final S3 key is composed of:
```
{s3_prefix}/{file_name_prefix}_{timestamp}.{extension}
```
For example, with `s3_prefix="logs/detections"`, `file_name_prefix="run"`, and `file_type="csv"`:
```
logs/detections/run_2024_10_18_14_09_57_622297.csv
```
If `s3_prefix` is empty, the key starts directly with the file name.

## Note on Append Log Mode

In append log mode, data is buffered in memory and only uploaded to S3 when:
- The `max_entries_per_file` limit is reached (object rotation), or
- The block instance is destroyed at workflow teardown

This means data may not be immediately visible in S3 after each step execution. Use `separate_files` mode if immediate S3 visibility is required.

## Common Use Cases

- **Cloud Data Logging**: Upload detection results, metrics, or workflow outputs directly to S3 for durable cloud storage and downstream processing
- **Data Pipeline Integration**: Export formatted CSV or JSONL files to S3 for consumption by data pipelines, analytics tools, or ML training jobs
- **Batch Result Archival**: Store individual inference results as separate S3 objects organized by timestamp and prefix
- **Time-Series Collection**: Aggregate workflow outputs into batched JSONL or CSV files in S3 for cost-efficient log storage
- **Cross-Service Integration**: Write data to S3 to trigger Lambda functions, feed SQS queues, or integrate with other AWS services


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/s3_sink@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `file_type` | `str` | Type of file to create: 'csv' (CSV format), 'json' (JSON format, or JSONL in append_log mode), or 'txt' (plain text). In append_log mode, JSON files are stored as .jsonl (JSON Lines) format with one JSON object per line.. | ❌ |
| `output_mode` | `str` | Upload strategy: 'append_log' buffers multiple entries and uploads them as a single S3 object when the entry limit is reached (useful for batched logging), or 'separate_files' uploads each input as a new S3 object with a unique timestamp-based key (useful for per-execution outputs).. | ❌ |
| `bucket_name` | `str` | Name of the target S3 bucket. Can be a static string or a selector resolving to a string at runtime.. | ✅ |
| `s3_prefix` | `str` | S3 key prefix (folder path) where objects will be stored. Trailing slashes are normalized automatically. Combined with file_name_prefix and a timestamp to form the full object key. Example: 'logs/detections' produces keys like 'logs/detections/workflow_output_2024_10_18_14_09_57_622297.csv'.. | ✅ |
| `file_name_prefix` | `str` | Prefix used to generate S3 object names. Combined with a timestamp (format: YYYY_MM_DD_HH_MM_SS_microseconds) and file extension to create unique keys like 'workflow_output_2024_10_18_14_09_57_622297.csv'.. | ✅ |
| `max_entries_per_file` | `int` | Maximum number of buffered entries before uploading to S3 and starting a new object in append_log mode. When this limit is reached, the accumulated buffer is uploaded as a complete S3 object and a new buffer starts with a fresh key. Only applies when output_mode is 'append_log'. Must be at least 1.. | ✅ |
| `aws_access_key_id` | `str` | AWS access key ID for authentication. If not provided, boto3's default credential chain is used (environment variables, ~/.aws/credentials, or IAM role). Recommended: connect this to an Environment Secrets Store block rather than hardcoding.. | ✅ |
| `aws_secret_access_key` | `str` | AWS secret access key for authentication. If not provided, boto3's default credential chain is used. Recommended: connect this to an Environment Secrets Store block rather than hardcoding.. | ✅ |
| `aws_region` | `str` | AWS region where the bucket is located (e.g., 'us-east-1'). If not provided, boto3's default region is used (AWS_DEFAULT_REGION environment variable or ~/.aws/config).. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-alert-circle-outline:{ style="color: #f57c00" } `soft` — runtime `hosted_serverless`, `dedicated_deployment`; execution `remote`
:   Append-log mode buffers entries in process memory before uploading the accumulated object to S3. With remote step execution on stateless or multi-replica HTTP runtimes, successive requests may be served by different worker processes, so append-log objects can reset or split across workers. Use separate_files mode, or local step execution in an InferencePipeline when each entry must be captured in a single ordered log.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `S3 Sink` in version `v1`.

    - inputs: [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Florence-2 Model`](florence2_model.md), [`Email Notification`](email_notification.md), [`VLM As Classifier`](vlm_as_classifier.md), [`EasyOCR`](easy_ocr.md), [`Object Detection Model`](object_detection_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Local File Sink`](local_file_sink.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Qwen-VL`](qwen_vl.md), [`OpenAI`](open_ai.md), [`CogVLM`](cog_vlm.md), [`Google Vision OCR`](google_vision_ocr.md), [`OpenRouter`](open_router.md), [`Florence-2 Model`](florence2_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`OpenAI`](open_ai.md), [`S3 Sink`](s3_sink.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Google Gemini`](google_gemini.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Webhook Sink`](webhook_sink.md), [`OCR Model`](ocr_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`LMM For Classification`](lmm_for_classification.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Anthropic Claude`](anthropic_claude.md), [`Clip Comparison`](clip_comparison.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Current Time`](current_time.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Anthropic Claude`](anthropic_claude.md), [`MQTT Writer`](mqtt_writer.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemini`](google_gemini.md), [`Google Gemma`](google_gemma.md), [`Google Gemma API`](google_gemma_api.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Slack Notification`](slack_notification.md), [`Email Notification`](email_notification.md), [`LMM`](lmm.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Event Writer`](event_writer.md), [`GLM-OCR`](glmocr.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`VLM As Detector`](vlm_as_detector.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`CSV Formatter`](csv_formatter.md), [`OpenAI`](open_ai.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Anthropic Claude`](anthropic_claude.md)
    - outputs: [`Detections Classes Replacement`](detections_classes_replacement.md), [`Morphological Transformation`](morphological_transformation.md), [`Image Preprocessing`](image_preprocessing.md), [`Email Notification`](email_notification.md), [`Halo Visualization`](halo_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Object Detection Model`](object_detection_model.md), [`Time in Zone`](timein_zone.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Text Display`](text_display.md), [`Template Matching`](template_matching.md), [`Pixel Color Count`](pixel_color_count.md), [`Image Threshold`](image_threshold.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Time in Zone`](timein_zone.md), [`Qwen-VL`](qwen_vl.md), [`OpenAI`](open_ai.md), [`CogVLM`](cog_vlm.md), [`Crop Visualization`](crop_visualization.md), [`SAM 3`](sam3.md), [`Dot Visualization`](dot_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`Florence-2 Model`](florence2_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`S3 Sink`](s3_sink.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`QR Code Generator`](qr_code_generator.md), [`SIFT Comparison`](sift_comparison.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Cache Get`](cache_get.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Dynamic Zone`](dynamic_zone.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Gaze Detection`](gaze_detection.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`LMM For Classification`](lmm_for_classification.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Blur`](image_blur.md), [`Object Detection Model`](object_detection_model.md), [`Blur Visualization`](blur_visualization.md), [`Path Deviation`](path_deviation.md), [`Current Time`](current_time.md), [`SAM 3`](sam3.md), [`Perspective Correction`](perspective_correction.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`MQTT Writer`](mqtt_writer.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemini`](google_gemini.md), [`SAM 3`](sam3.md), [`Depth Estimation`](depth_estimation.md), [`Detections Consensus`](detections_consensus.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Google Gemma API`](google_gemma_api.md), [`Object Detection Model`](object_detection_model.md), [`Slack Notification`](slack_notification.md), [`Time in Zone`](timein_zone.md), [`Image Stack`](image_stack.md), [`Google Gemini`](google_gemini.md), [`Cache Set`](cache_set.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Label Visualization`](label_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Size Measurement`](size_measurement.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI`](open_ai.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Moondream2`](moondream2.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Florence-2 Model`](florence2_model.md), [`Seg Preview`](seg_preview.md), [`YOLO-World Model`](yolo_world_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Local File Sink`](local_file_sink.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Triangle Visualization`](triangle_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Path Deviation`](path_deviation.md), [`OpenRouter`](open_router.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Distance Measurement`](distance_measurement.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OpenAI`](open_ai.md), [`Background Color Visualization`](background_color_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemini`](google_gemini.md), [`Corner Visualization`](corner_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Line Counter`](line_counter.md), [`Halo Visualization`](halo_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Dynamic Crop`](dynamic_crop.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Motion Detection`](motion_detection.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Camera Calibration`](camera_calibration.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Google Gemma`](google_gemma.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Line Counter`](line_counter.md), [`Circle Visualization`](circle_visualization.md), [`Email Notification`](email_notification.md), [`LMM`](lmm.md), [`Event Writer`](event_writer.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`GLM-OCR`](glmocr.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Mask Visualization`](mask_visualization.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Polygon Visualization`](polygon_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`S3 Sink` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `content` (*[`string`](../kinds/string.md)*): String content to upload to S3. This should be formatted data from other workflow blocks (e.g., CSV content from CSV Formatter, JSON strings, or plain text). The content format should match the specified file_type. For CSV files in append_log mode, content must include header rows on the first write..
        - `bucket_name` (*[`string`](../kinds/string.md)*): Name of the target S3 bucket. Can be a static string or a selector resolving to a string at runtime..
        - `s3_prefix` (*[`string`](../kinds/string.md)*): S3 key prefix (folder path) where objects will be stored. Trailing slashes are normalized automatically. Combined with file_name_prefix and a timestamp to form the full object key. Example: 'logs/detections' produces keys like 'logs/detections/workflow_output_2024_10_18_14_09_57_622297.csv'..
        - `file_name_prefix` (*[`string`](../kinds/string.md)*): Prefix used to generate S3 object names. Combined with a timestamp (format: YYYY_MM_DD_HH_MM_SS_microseconds) and file extension to create unique keys like 'workflow_output_2024_10_18_14_09_57_622297.csv'..
        - `max_entries_per_file` (*[`string`](../kinds/string.md)*): Maximum number of buffered entries before uploading to S3 and starting a new object in append_log mode. When this limit is reached, the accumulated buffer is uploaded as a complete S3 object and a new buffer starts with a fresh key. Only applies when output_mode is 'append_log'. Must be at least 1..
        - `aws_access_key_id` (*Union[[`string`](../kinds/string.md), [`secret`](../kinds/secret.md)]*): AWS access key ID for authentication. If not provided, boto3's default credential chain is used (environment variables, ~/.aws/credentials, or IAM role). Recommended: connect this to an Environment Secrets Store block rather than hardcoding..
        - `aws_secret_access_key` (*Union[[`string`](../kinds/string.md), [`secret`](../kinds/secret.md)]*): AWS secret access key for authentication. If not provided, boto3's default credential chain is used. Recommended: connect this to an Environment Secrets Store block rather than hardcoding..
        - `aws_region` (*[`string`](../kinds/string.md)*): AWS region where the bucket is located (e.g., 'us-east-1'). If not provided, boto3's default region is used (AWS_DEFAULT_REGION environment variable or ~/.aws/config)..

    - output
    
        - `error_status` ([`boolean`](../kinds/boolean.md)): Boolean flag.
        - `message` ([`string`](../kinds/string.md)): String value.



??? tip "Example JSON definition of step `S3 Sink` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/s3_sink@v1",
	    "content": "$steps.csv_formatter.csv_content",
	    "file_type": "csv",
	    "output_mode": "append_log",
	    "bucket_name": "my-inference-results",
	    "s3_prefix": "logs/detections",
	    "file_name_prefix": "my_output",
	    "max_entries_per_file": 1024,
	    "aws_access_key_id": "$steps.secrets.aws_access_key_id",
	    "aws_secret_access_key": "$steps.secrets.aws_secret_access_key",
	    "aws_region": "us-east-1"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    


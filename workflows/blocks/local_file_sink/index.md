
# Local File Sink



??? "Class: `LocalFileSinkBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/sinks/local_file/v1.py">inference.core.workflows.core_steps.sinks.local_file.v1.LocalFileSinkBlockV1</a>
    



Save workflow data as files on the local filesystem, supporting CSV, JSON, and text file formats with configurable output modes for aggregating multiple entries into single files or saving each entry separately, enabling persistent data storage, logging, and file-based data export.

## How This Block Works

This block writes string content from workflow steps to files on the local filesystem. The block:

1. Takes string content (from formatters, predictions, or other string-producing blocks) and file configuration as input
2. Validates filesystem access permissions (checks if local storage access is allowed based on environment configuration)
3. Verifies write permissions for the target directory (checks against allowed write directory restrictions if configured)
4. Selects the appropriate file saving strategy based on `output_mode`:
   - **Separate Files Mode**: Creates a new file for each input, generating unique filenames with timestamps
   - **Append Log Mode**: Appends content to an existing file (or creates a new one if needed), aggregating multiple entries
5. For **separate files mode**: Generates a unique file path using the target directory, file name prefix, file type, and a timestamp, then writes the content to the new file
6. For **append log mode**: 
   - Opens or creates a file based on the file name prefix and type
   - Applies format-specific handling for appending:
     - **CSV**: Removes the header row from subsequent appends (CSV content must include headers on first write)
     - **JSON**: Converts to JSONL (JSON Lines) format, parsing and re-serializing each JSON document to fit on a single line
     - **TXT**: Appends content directly with newlines
   - Tracks entry count and creates a new file when `max_entries_per_file` limit is reached
7. Creates parent directories if they don't exist
8. Writes content to the file (ensuring newline termination)
9. Returns error status and messages indicating save success or failure

The block supports two distinct storage strategies: separate files mode creates individual timestamped files for each input (useful for organizing outputs by execution), while append log mode aggregates multiple entries into continuous log files (useful for time-series data logging). The file path generation includes timestamps (format: `YYYY_MM_DD_HH_MM_SS_microseconds`) to ensure unique filenames and chronological organization. In append log mode, the block maintains file handles across executions and automatically handles file rotation when entry limits are reached.

## Requirements

**Local Filesystem Access**: This block requires write access to the local filesystem. Filesystem access can be controlled via environment variables:
- Set `ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE=False` to disable local file sink functionality (block will raise an error)
- Set `WORKFLOW_BLOCKS_WRITE_DIRECTORY` to an absolute path to restrict writes to a specific directory and its subdirectories only

**Note on Append Log Mode Format Handling**: 
- For CSV files in append mode, the content must include header rows on the first write; headers are automatically removed from subsequent appends
- For JSON files in append mode, files are saved with `.jsonl` extension in JSON Lines format (one JSON object per line)

## Common Use Cases

- **Data Logging and Audit Trails**: Save workflow execution data, detection results, or metrics to local log files (e.g., append CSV logs of detections, JSON logs of workflow outputs), enabling persistent logging and audit trails for production workflows
- **File-Based Data Export**: Export formatted workflow data to files for external processing (e.g., save CSV exports from CSV Formatter, JSON exports for downstream tools), enabling integration with file-based data processing pipelines
- **Time-Series Data Collection**: Aggregate workflow metrics over time into continuous log files (e.g., append CSV rows with timestamps, log detection counts per frame), creating persistent time-series datasets for analysis and reporting
- **Batch Result Storage**: Save individual results from batch processing workflows to separate files (e.g., save each image's detection results to separate JSON files), enabling organized storage of batch processing outputs with unique filenames
- **Data Archival**: Archive workflow outputs and results to local storage (e.g., save formatted reports, export analysis results), enabling long-term data retention and backup workflows
- **Integration with File-Based Systems**: Store workflow data in file formats compatible with external tools (e.g., save CSV for spreadsheet analysis, JSONL for data processing pipelines), enabling seamless data exchange with file-based systems

## Connecting to Other Blocks

This block receives string content from workflow steps and saves it to files:

- **After formatter blocks** (e.g., CSV Formatter) to save formatted data (CSV, JSON, or text) to files, enabling persistent storage of structured workflow outputs
- **After detection or analysis blocks** that output string-format data to save inference results, metrics, or analysis outputs to files for logging or archival
- **After data processing blocks** (e.g., Expression, Property Definition) that produce string outputs to save computed or transformed data to files
- **In logging workflows** to create persistent audit trails and logs of workflow executions, enabling record-keeping and debugging for production deployments
- **In batch processing workflows** where multiple data points need to be saved (either aggregated into log files or stored as separate files), enabling organized data collection and storage
- **Before external processing** where workflow data needs to be saved to files for consumption by external tools, scripts, or systems that read from filesystem storage


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/local_file_sink@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `file_type` | `str` | Type of file to create: 'csv' (CSV format), 'json' (JSON format, or JSONL in append_log mode), or 'txt' (plain text). The content format should match this file type. In append_log mode, JSON files are saved as .jsonl (JSON Lines) format with one JSON object per line.. | ❌ |
| `output_mode` | `str` | File organization strategy: 'append_log' aggregates multiple content entries into a single file (useful for time-series logging, creates files that grow over time), or 'separate_files' creates a new file for each input (useful for organizing individual outputs, each file gets a unique timestamp-based filename). In append_log mode, the block handles format-specific appending (removes CSV headers, converts JSON to JSONL).. | ❌ |
| `target_directory` | `str` | Directory path where files will be saved. Can be a relative or absolute path. Parent directories are created automatically if they don't exist. If WORKFLOW_BLOCKS_WRITE_DIRECTORY is set, this path must be a subdirectory of the allowed directory. Files are saved with filenames generated from file_name_prefix and timestamps.. | ✅ |
| `file_name_prefix` | `str` | Prefix used to generate filenames. Combined with a timestamp (format: YYYY_MM_DD_HH_MM_SS_microseconds) and file extension to create unique filenames like 'workflow_output_2024_10_18_14_09_57_622297.csv'. For append_log mode, new files are created when max_entries_per_file is reached, using this prefix with new timestamps.. | ✅ |
| `max_entries_per_file` | `int` | Maximum number of entries (content appends) allowed per file in append_log mode. When this limit is reached, a new file is created with the same file_name_prefix and a new timestamp. Only applies when output_mode is 'append_log'. Must be at least 1. Use this to control file sizes and enable file rotation for long-running workflows.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-alert-circle-outline:{ style="color: #f57c00" } `soft` — runtime `dedicated_deployment`
:   Files are persisted on the deployment's volume but are not retrievable through the Roboflow API; treat as internal-only logs.

:material-alert-circle-outline:{ style="color: #f57c00" } `soft` — runtime `hosted_serverless`
:   Container disk is ephemeral, so files are lost when the worker scales down; if there's more than one replica consuming workflow requests the result will be non deterministic..

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Local File Sink` in version `v1`.

    - inputs: [`Google Gemma API`](google_gemma_api.md), [`Event Writer`](event_writer.md), [`Google Gemini`](google_gemini.md), [`Current Time`](current_time.md), [`Email Notification`](email_notification.md), [`Local File Sink`](local_file_sink.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Google Vision OCR`](google_vision_ocr.md), [`Object Detection Model`](object_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Anthropic Claude`](anthropic_claude.md), [`OpenRouter`](open_router.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Gemini`](google_gemini.md), [`EasyOCR`](easy_ocr.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`LMM`](lmm.md), [`VLM As Detector`](vlm_as_detector.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Google Gemma`](google_gemma.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`MQTT Writer`](mqtt_writer.md), [`Anthropic Claude`](anthropic_claude.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`CSV Formatter`](csv_formatter.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`LMM For Classification`](lmm_for_classification.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`CogVLM`](cog_vlm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Florence-2 Model`](florence2_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Clip Comparison`](clip_comparison.md), [`Qwen-VL`](qwen_vl.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`OCR Model`](ocr_model.md), [`OpenAI`](open_ai.md), [`Email Notification`](email_notification.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Slack Notification`](slack_notification.md), [`S3 Sink`](s3_sink.md), [`GLM-OCR`](glmocr.md), [`Webhook Sink`](webhook_sink.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`OPC UA Writer Sink`](opcua_writer_sink.md)
    - outputs: [`Google Gemma API`](google_gemma_api.md), [`Event Writer`](event_writer.md), [`Google Gemini`](google_gemini.md), [`Current Time`](current_time.md), [`Ellipse Visualization`](ellipse_visualization.md), [`SAM 3`](sam3.md), [`Image Preprocessing`](image_preprocessing.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Color Visualization`](color_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`YOLO-World Model`](yolo_world_model.md), [`Seg Preview`](seg_preview.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Cache Set`](cache_set.md), [`Google Vision OCR`](google_vision_ocr.md), [`SIFT Comparison`](sift_comparison.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Object Detection Model`](object_detection_model.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`SAM 3`](sam3.md), [`Google Gemma`](google_gemma.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Blur`](image_blur.md), [`Keypoint Visualization`](keypoint_visualization.md), [`MQTT Writer`](mqtt_writer.md), [`Anthropic Claude`](anthropic_claude.md), [`Camera Calibration`](camera_calibration.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Detections Stitch`](detections_stitch.md), [`LMM For Classification`](lmm_for_classification.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Perspective Correction`](perspective_correction.md), [`Cache Get`](cache_get.md), [`Time in Zone`](timein_zone.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`SAM 3`](sam3.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Contrast Equalization`](contrast_equalization.md), [`Clip Comparison`](clip_comparison.md), [`Qwen-VL`](qwen_vl.md), [`Image Stack`](image_stack.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Dot Visualization`](dot_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`OpenAI`](open_ai.md), [`Moondream2`](moondream2.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Size Measurement`](size_measurement.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Path Deviation`](path_deviation.md), [`Webhook Sink`](webhook_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Halo Visualization`](halo_visualization.md), [`Email Notification`](email_notification.md), [`QR Code Generator`](qr_code_generator.md), [`Local File Sink`](local_file_sink.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Object Detection Model`](object_detection_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Path Deviation`](path_deviation.md), [`Template Matching`](template_matching.md), [`Depth Estimation`](depth_estimation.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Text Display`](text_display.md), [`Time in Zone`](timein_zone.md), [`Object Detection Model`](object_detection_model.md), [`Distance Measurement`](distance_measurement.md), [`Icon Visualization`](icon_visualization.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`OpenRouter`](open_router.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Anthropic Claude`](anthropic_claude.md), [`LMM`](lmm.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Image Threshold`](image_threshold.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Label Visualization`](label_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Morphological Transformation`](morphological_transformation.md), [`Morphological Transformation`](morphological_transformation.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Gaze Detection`](gaze_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`CogVLM`](cog_vlm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Florence-2 Model`](florence2_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Background Color Visualization`](background_color_visualization.md), [`Detections Consensus`](detections_consensus.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Corner Visualization`](corner_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Line Counter Visualization`](line_counter_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Halo Visualization`](halo_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Line Counter`](line_counter.md), [`OpenAI`](open_ai.md), [`Email Notification`](email_notification.md), [`Slack Notification`](slack_notification.md), [`S3 Sink`](s3_sink.md), [`GLM-OCR`](glmocr.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OPC UA Writer Sink`](opcua_writer_sink.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Local File Sink` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `content` (*[`string`](../kinds/string.md)*): String content to save as a file. This should be formatted data from other workflow blocks (e.g., CSV content from CSV Formatter, JSON strings, or plain text). The content format should match the specified file_type. For CSV files in append_log mode, content must include header rows on the first write..
        - `target_directory` (*[`string`](../kinds/string.md)*): Directory path where files will be saved. Can be a relative or absolute path. Parent directories are created automatically if they don't exist. If WORKFLOW_BLOCKS_WRITE_DIRECTORY is set, this path must be a subdirectory of the allowed directory. Files are saved with filenames generated from file_name_prefix and timestamps..
        - `file_name_prefix` (*[`string`](../kinds/string.md)*): Prefix used to generate filenames. Combined with a timestamp (format: YYYY_MM_DD_HH_MM_SS_microseconds) and file extension to create unique filenames like 'workflow_output_2024_10_18_14_09_57_622297.csv'. For append_log mode, new files are created when max_entries_per_file is reached, using this prefix with new timestamps..
        - `max_entries_per_file` (*[`string`](../kinds/string.md)*): Maximum number of entries (content appends) allowed per file in append_log mode. When this limit is reached, a new file is created with the same file_name_prefix and a new timestamp. Only applies when output_mode is 'append_log'. Must be at least 1. Use this to control file sizes and enable file rotation for long-running workflows..

    - output
    
        - `error_status` ([`boolean`](../kinds/boolean.md)): Boolean flag.
        - `message` ([`string`](../kinds/string.md)): String value.



??? tip "Example JSON definition of step `Local File Sink` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/local_file_sink@v1",
	    "content": "$steps.csv_formatter.csv_content",
	    "file_type": "csv",
	    "output_mode": "append_log",
	    "target_directory": "some/location",
	    "file_name_prefix": "my_file",
	    "max_entries_per_file": 1024
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    


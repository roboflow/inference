
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

    - inputs: [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Florence-2 Model`](florence2_model.md), [`Email Notification`](email_notification.md), [`VLM As Classifier`](vlm_as_classifier.md), [`EasyOCR`](easy_ocr.md), [`Object Detection Model`](object_detection_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Local File Sink`](local_file_sink.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Qwen-VL`](qwen_vl.md), [`OpenAI`](open_ai.md), [`CogVLM`](cog_vlm.md), [`Google Vision OCR`](google_vision_ocr.md), [`OpenRouter`](open_router.md), [`Florence-2 Model`](florence2_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`OpenAI`](open_ai.md), [`S3 Sink`](s3_sink.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Google Gemini`](google_gemini.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Webhook Sink`](webhook_sink.md), [`OCR Model`](ocr_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`LMM For Classification`](lmm_for_classification.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Anthropic Claude`](anthropic_claude.md), [`Clip Comparison`](clip_comparison.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Current Time`](current_time.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Anthropic Claude`](anthropic_claude.md), [`MQTT Writer`](mqtt_writer.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemini`](google_gemini.md), [`Google Gemma`](google_gemma.md), [`Google Gemma API`](google_gemma_api.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Slack Notification`](slack_notification.md), [`Email Notification`](email_notification.md), [`LMM`](lmm.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Event Writer`](event_writer.md), [`GLM-OCR`](glmocr.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`VLM As Detector`](vlm_as_detector.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`CSV Formatter`](csv_formatter.md), [`OpenAI`](open_ai.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Anthropic Claude`](anthropic_claude.md)
    - outputs: [`Detections Classes Replacement`](detections_classes_replacement.md), [`Morphological Transformation`](morphological_transformation.md), [`Image Preprocessing`](image_preprocessing.md), [`Email Notification`](email_notification.md), [`Halo Visualization`](halo_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Object Detection Model`](object_detection_model.md), [`Time in Zone`](timein_zone.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Text Display`](text_display.md), [`Template Matching`](template_matching.md), [`Pixel Color Count`](pixel_color_count.md), [`Image Threshold`](image_threshold.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Time in Zone`](timein_zone.md), [`Qwen-VL`](qwen_vl.md), [`OpenAI`](open_ai.md), [`CogVLM`](cog_vlm.md), [`Crop Visualization`](crop_visualization.md), [`SAM 3`](sam3.md), [`Dot Visualization`](dot_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`Florence-2 Model`](florence2_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`S3 Sink`](s3_sink.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`QR Code Generator`](qr_code_generator.md), [`SIFT Comparison`](sift_comparison.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Cache Get`](cache_get.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Dynamic Zone`](dynamic_zone.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Gaze Detection`](gaze_detection.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`LMM For Classification`](lmm_for_classification.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Blur`](image_blur.md), [`Object Detection Model`](object_detection_model.md), [`Blur Visualization`](blur_visualization.md), [`Path Deviation`](path_deviation.md), [`Current Time`](current_time.md), [`SAM 3`](sam3.md), [`Perspective Correction`](perspective_correction.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`MQTT Writer`](mqtt_writer.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemini`](google_gemini.md), [`SAM 3`](sam3.md), [`Depth Estimation`](depth_estimation.md), [`Detections Consensus`](detections_consensus.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Google Gemma API`](google_gemma_api.md), [`Object Detection Model`](object_detection_model.md), [`Slack Notification`](slack_notification.md), [`Time in Zone`](timein_zone.md), [`Image Stack`](image_stack.md), [`Google Gemini`](google_gemini.md), [`Cache Set`](cache_set.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Label Visualization`](label_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Size Measurement`](size_measurement.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI`](open_ai.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Moondream2`](moondream2.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Florence-2 Model`](florence2_model.md), [`Seg Preview`](seg_preview.md), [`YOLO-World Model`](yolo_world_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Local File Sink`](local_file_sink.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Triangle Visualization`](triangle_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Path Deviation`](path_deviation.md), [`OpenRouter`](open_router.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Distance Measurement`](distance_measurement.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OpenAI`](open_ai.md), [`Background Color Visualization`](background_color_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemini`](google_gemini.md), [`Corner Visualization`](corner_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Line Counter`](line_counter.md), [`Halo Visualization`](halo_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Dynamic Crop`](dynamic_crop.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Motion Detection`](motion_detection.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Camera Calibration`](camera_calibration.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Google Gemma`](google_gemma.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Line Counter`](line_counter.md), [`Circle Visualization`](circle_visualization.md), [`Email Notification`](email_notification.md), [`LMM`](lmm.md), [`Event Writer`](event_writer.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`GLM-OCR`](glmocr.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Mask Visualization`](mask_visualization.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Polygon Visualization`](polygon_visualization.md)

    
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


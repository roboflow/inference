
# JSON Parser



??? "Class: `JSONParserBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/formatters/json_parser/v1.py">inference.core.workflows.core_steps.formatters.json_parser.v1.JSONParserBlockV1</a>
    



Parse JSON strings (raw JSON or JSON wrapped in Markdown code blocks) into structured data by extracting specified fields and exposing them as individual outputs, enabling JSON parsing, LLM/VLM output processing, structured data extraction, and configuration parsing workflows where JSON strings need to be converted into usable workflow data.

## How This Block Works

This block parses JSON strings and extracts specified fields as individual outputs. The block:

1. Receives a JSON string input (typically from LLM/VLM blocks or workflow inputs)
2. Detects and extracts JSON content:

   **Handles Markdown-wrapped JSON:**
   - Searches for JSON wrapped in Markdown code blocks (```json ... ```)
   - This format is very common in LLM/VLM responses (e.g., GPT responses)
   - If multiple markdown JSON blocks are found, only the first block is parsed
   - Extracts the JSON content from within the markdown tags

   **Handles raw JSON strings:**
   - If no markdown blocks are found, attempts to parse the entire string as JSON
   - Supports standard JSON format strings
3. Parses JSON content:
   - Uses Python's JSON parser to convert the string into a JSON object/dictionary
   - Handles parsing errors gracefully (returns None for all fields if parsing fails)
4. Extracts expected fields:
   - Retrieves values for each field specified in `expected_fields` parameter
   - For each expected field, looks up the corresponding key in the parsed JSON
   - Returns the field value (or None if the field is missing)
5. Sets error status:
   - `error_status` is set to `True` if at least one expected field cannot be retrieved from the parsed JSON
   - `error_status` is set to `False` if all expected fields are found (even if multiple markdown blocks exist, only first is parsed)
   - Error status is always included as an output, allowing downstream blocks to check parsing success
6. Exposes fields as outputs:
   - Each field in `expected_fields` becomes a separate output with the field name
   - Field values are extracted from the parsed JSON and made available as outputs
   - Missing fields are set to None
   - All outputs can be referenced using `$steps.block_name.field_name` syntax
7. Returns parsed data:
   - Outputs include: `error_status` (boolean) and all expected fields
   - Fields contain the extracted values from the JSON (or None if missing)
   - Outputs can be used in subsequent workflow steps

The block is particularly useful for processing LLM/VLM outputs that return JSON, extracting structured configuration from JSON strings, and parsing JSON responses into workflow-usable data. It handles the common case where LLMs wrap JSON in markdown code blocks.

## Common Use Cases

- **LLM/VLM Output Processing**: Parse JSON outputs from Large Language Models and Visual Language Models (e.g., parse GPT JSON responses, extract structured data from LLM outputs, process VLM JSON responses), enabling LLM/VLM output processing workflows
- **Structured Data Extraction**: Extract structured data from JSON strings for use in workflows (e.g., extract configuration parameters, parse JSON responses, extract structured fields), enabling structured data extraction workflows
- **Configuration Parsing**: Parse JSON configuration strings into workflow parameters (e.g., parse model configuration, extract workflow parameters, parse JSON configs), enabling configuration parsing workflows
- **JSON Response Processing**: Process JSON responses from APIs or models (e.g., parse API JSON responses, extract fields from JSON, process JSON data), enabling JSON response processing workflows
- **Dynamic Parameter Extraction**: Extract dynamic parameters from JSON strings for use in workflow steps (e.g., extract model IDs from JSON, parse dynamic configs, extract parameters dynamically), enabling dynamic parameter workflows
- **Data Format Conversion**: Convert JSON strings into structured workflow data (e.g., convert JSON to workflow inputs, parse JSON for workflow use, extract JSON fields), enabling data format conversion workflows

## Connecting to Other Blocks

This block receives JSON strings and produces parsed field outputs:

- **After LLM/VLM blocks** to parse JSON outputs into structured data (e.g., parse LLM JSON outputs, extract VLM JSON fields, process model JSON responses), enabling LLM/VLM-to-parser workflows
- **After workflow inputs** to parse JSON input parameters (e.g., parse JSON config inputs, extract JSON parameters, process JSON workflow inputs), enabling input-parser workflows
- **Before model blocks** to use parsed fields as model parameters (e.g., use parsed model_id for models, use parsed configs for model setup, provide parsed parameters to models), enabling parser-to-model workflows
- **Before logic blocks** to use parsed fields in conditions (e.g., use parsed values in Continue If, filter based on parsed fields, make decisions using parsed data), enabling parser-to-logic workflows
- **Before data storage blocks** to store parsed field values (e.g., store parsed JSON fields, log parsed values, save parsed data), enabling parser-to-storage workflows
- **In workflow outputs** to provide parsed fields as final output (e.g., JSON parsing outputs, structured data outputs, parsed field outputs), enabling parser-to-output workflows

## Requirements

This block requires a JSON string input (raw JSON or JSON wrapped in Markdown code blocks). The `expected_fields` parameter specifies which JSON fields to extract as outputs (field names must be valid JSON keys). The `error_status` field name is reserved and cannot be used in `expected_fields`. The block supports both raw JSON strings and JSON wrapped in markdown code blocks (```json ... ```). If multiple markdown blocks are found, only the first is parsed. If parsing fails or expected fields are missing, fields are set to None and `error_status` is set to True. All expected fields become separate outputs that can be referenced in subsequent workflow steps.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/json_parser@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `raw_json` | `str` | JSON string to parse. Can be raw JSON string (e.g., '{"key": "value"}') or JSON wrapped in Markdown code blocks (e.g., ```json {"key": "value"} ```). Markdown-wrapped JSON is common in LLM/VLM responses. If multiple markdown JSON blocks are present, only the first block is parsed. The string is parsed using Python's JSON parser, and specified fields are extracted as outputs.. | ✅ |
| `expected_fields` | `List[str]` | List of JSON field names to extract from the parsed JSON. Each field becomes a separate output that can be referenced in subsequent workflow steps (e.g., $steps.block_name.field_name). Fields that exist in the JSON are extracted with their values; missing fields are set to None. The 'error_status' field name is reserved (always included as output) and cannot be used in this list. Field names must match JSON keys exactly.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `JSON Parser` in version `v1`.

    - inputs: [`Llama 3.2 Vision`](llama3.2_vision.md), [`Google Gemini`](google_gemini.md), [`Anthropic Claude`](anthropic_claude.md), [`Anthropic Claude`](anthropic_claude.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenAI`](open_ai.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Google Gemini`](google_gemini.md), [`GLM-OCR`](glmocr.md), [`Qwen-VL`](qwen_vl.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`OpenRouter`](open_router.md), [`Florence-2 Model`](florence2_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Google Gemma`](google_gemma.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenAI`](open_ai.md), [`Google Gemma API`](google_gemma_api.md), [`Anthropic Claude`](anthropic_claude.md)
    - outputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Qwen-VL`](qwen_vl.md), [`Slack Notification`](slack_notification.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Mask Edge Snap`](mask_edge_snap.md), [`VLM As Detector`](vlm_as_detector.md), [`Template Matching`](template_matching.md), [`Dynamic Zone`](dynamic_zone.md), [`Track Class Lock`](track_class_lock.md), [`Webhook Sink`](webhook_sink.md), [`SAM 3`](sam3.md), [`Contrast Enhancement`](contrast_enhancement.md), [`QR Code Generator`](qr_code_generator.md), [`SIFT`](sift.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Florence-2 Model`](florence2_model.md), [`Path Deviation`](path_deviation.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Delta Filter`](delta_filter.md), [`Text Display`](text_display.md), [`MQTT Writer`](mqtt_writer.md), [`Gaze Detection`](gaze_detection.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Expression`](expression.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Time in Zone`](timein_zone.md), [`Time in Zone`](timein_zone.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Image Slicer`](image_slicer.md), [`Background Subtraction`](background_subtraction.md), [`Image Preprocessing`](image_preprocessing.md), [`Byte Tracker`](byte_tracker.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Email Notification`](email_notification.md), [`Line Counter`](line_counter.md), [`LMM`](lmm.md), [`Distance Measurement`](distance_measurement.md), [`Detections Consensus`](detections_consensus.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`SAM 3`](sam3.md), [`Time in Zone`](timein_zone.md), [`Event Writer`](event_writer.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Image Blur`](image_blur.md), [`Google Vision OCR`](google_vision_ocr.md), [`Byte Tracker`](byte_tracker.md), [`Qwen3-VL`](qwen3_vl.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`CSV Formatter`](csv_formatter.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Detections Merge`](detections_merge.md), [`OpenAI`](open_ai.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`Detections Transformation`](detections_transformation.md), [`LMM For Classification`](lmm_for_classification.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Color Visualization`](color_visualization.md), [`Property Definition`](property_definition.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Object Detection Model`](object_detection_model.md), [`Cache Get`](cache_get.md), [`Seg Preview`](seg_preview.md), [`Camera Calibration`](camera_calibration.md), [`Dominant Color`](dominant_color.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Detections Filter`](detections_filter.md), [`Background Color Visualization`](background_color_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`CogVLM`](cog_vlm.md), [`SORT Tracker`](sort_tracker.md), [`OpenAI`](open_ai.md), [`Identify Outliers`](identify_outliers.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Depth Estimation`](depth_estimation.md), [`Image Stack`](image_stack.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Google Gemma`](google_gemma.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Detections Stitch`](detections_stitch.md), [`Buffer`](buffer.md), [`Crop Visualization`](crop_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Camera Focus`](camera_focus.md), [`JSON Parser`](json_parser.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`GLM-OCR`](glmocr.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Dimension Collapse`](dimension_collapse.md), [`Continue If`](continue_if.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`VLM As Detector`](vlm_as_detector.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Threshold`](image_threshold.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen3.5`](qwen3.5.md), [`OpenAI`](open_ai.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Detections Combine`](detections_combine.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Cosine Similarity`](cosine_similarity.md), [`Florence-2 Model`](florence2_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Halo Visualization`](halo_visualization.md), [`SAM 3`](sam3.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Gemini`](google_gemini.md), [`YOLO-World Model`](yolo_world_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Halo Visualization`](halo_visualization.md), [`Label Visualization`](label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Byte Tracker`](byte_tracker.md), [`Inner Workflow`](inner_workflow.md), [`Clip Comparison`](clip_comparison.md), [`Detection Offset`](detection_offset.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Morphological Transformation`](morphological_transformation.md), [`PLC Writer`](plc_writer.md), [`Google Gemini`](google_gemini.md), [`PLC Reader`](plc_reader.md), [`Polygon Visualization`](polygon_visualization.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Moondream2`](moondream2.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Google Gemini`](google_gemini.md), [`EasyOCR`](easy_ocr.md), [`Overlap Analysis`](overlap_analysis.md), [`Motion Detection`](motion_detection.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Switch Case`](switch_case.md), [`Data Aggregator`](data_aggregator.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Local File Sink`](local_file_sink.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Rate Limiter`](rate_limiter.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Slicer`](image_slicer.md), [`Identify Changes`](identify_changes.md), [`S3 Sink`](s3_sink.md), [`Cache Set`](cache_set.md), [`Size Measurement`](size_measurement.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Path Deviation`](path_deviation.md), [`Contrast Equalization`](contrast_equalization.md), [`Morphological Transformation`](morphological_transformation.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Barcode Detection`](barcode_detection.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SmolVLM2`](smol_vlm2.md), [`Email Notification`](email_notification.md), [`OpenRouter`](open_router.md), [`Grid Visualization`](grid_visualization.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Velocity`](velocity.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Clip Comparison`](clip_comparison.md), [`Overlap Filter`](overlap_filter.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Corner Visualization`](corner_visualization.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Object Detection Model`](object_detection_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Line Counter`](line_counter.md), [`Current Time`](current_time.md), [`Stitch Images`](stitch_images.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Icon Visualization`](icon_visualization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`OCR Model`](ocr_model.md), [`Perspective Correction`](perspective_correction.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Pixel Color Count`](pixel_color_count.md), [`QR Code Detection`](qr_code_detection.md), [`Detection Event Log`](detection_event_log.md), [`Image Contours`](image_contours.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`JSON Parser` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `raw_json` (*[`language_model_output`](../kinds/language_model_output.md)*): JSON string to parse. Can be raw JSON string (e.g., '{"key": "value"}') or JSON wrapped in Markdown code blocks (e.g., ```json {"key": "value"} ```). Markdown-wrapped JSON is common in LLM/VLM responses. If multiple markdown JSON blocks are present, only the first block is parsed. The string is parsed using Python's JSON parser, and specified fields are extracted as outputs..

    - output
    
        - `error_status` ([`boolean`](../kinds/boolean.md)): Boolean flag.
        - `*` ([`*`](../kinds/wildcard.md)): Equivalent of any element.



??? tip "Example JSON definition of step `JSON Parser` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/json_parser@v1",
	    "raw_json": "$steps.lmm.output",
	    "expected_fields": [
	        "field_a",
	        "field_b"
	    ]
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    


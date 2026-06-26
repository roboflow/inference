
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

    - inputs: [`Anthropic Claude`](anthropic_claude.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`OpenAI`](open_ai.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Anthropic Claude`](anthropic_claude.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Gemini`](google_gemini.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`GLM-OCR`](glmocr.md), [`Google Gemma API`](google_gemma_api.md), [`Google Gemma`](google_gemma.md), [`OpenAI`](open_ai.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`OpenAI`](open_ai.md), [`Qwen-VL`](qwen_vl.md), [`OpenRouter`](open_router.md)
    - outputs: [`YOLO-World Model`](yolo_world_model.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Byte Tracker`](byte_tracker.md), [`SORT Tracker`](sort_tracker.md), [`Slack Notification`](slack_notification.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`LMM`](lmm.md), [`Qwen3-VL`](qwen3_vl.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Image Slicer`](image_slicer.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Anthropic Claude`](anthropic_claude.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Velocity`](velocity.md), [`Identify Outliers`](identify_outliers.md), [`Morphological Transformation`](morphological_transformation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`VLM As Detector`](vlm_as_detector.md), [`Background Subtraction`](background_subtraction.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`SAM 3`](sam3.md), [`CogVLM`](cog_vlm.md), [`Anthropic Claude`](anthropic_claude.md), [`QR Code Detection`](qr_code_detection.md), [`SIFT`](sift.md), [`Google Vision OCR`](google_vision_ocr.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`SIFT Comparison`](sift_comparison.md), [`Size Measurement`](size_measurement.md), [`PLC Writer`](plc_writer.md), [`Google Gemini`](google_gemini.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Stitch Images`](stitch_images.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Stack`](image_stack.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`QR Code Generator`](qr_code_generator.md), [`Dot Visualization`](dot_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Blur Visualization`](blur_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Dynamic Crop`](dynamic_crop.md), [`Halo Visualization`](halo_visualization.md), [`Rate Limiter`](rate_limiter.md), [`Local File Sink`](local_file_sink.md), [`Email Notification`](email_notification.md), [`Detections Transformation`](detections_transformation.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Florence-2 Model`](florence2_model.md), [`Cache Set`](cache_set.md), [`MQTT Writer`](mqtt_writer.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Time in Zone`](timein_zone.md), [`Time in Zone`](timein_zone.md), [`OCR Model`](ocr_model.md), [`Byte Tracker`](byte_tracker.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Depth Estimation`](depth_estimation.md), [`Google Gemini`](google_gemini.md), [`Time in Zone`](timein_zone.md), [`Track Class Lock`](track_class_lock.md), [`Camera Calibration`](camera_calibration.md), [`Text Display`](text_display.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Image Slicer`](image_slicer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemma API`](google_gemma_api.md), [`SAM 3`](sam3.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Path Deviation`](path_deviation.md), [`Dimension Collapse`](dimension_collapse.md), [`OpenRouter`](open_router.md), [`Detection Event Log`](detection_event_log.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Line Counter`](line_counter.md), [`Clip Comparison`](clip_comparison.md), [`Detections Stitch`](detections_stitch.md), [`Seg Preview`](seg_preview.md), [`Line Counter`](line_counter.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Object Detection Model`](object_detection_model.md), [`Event Writer`](event_writer.md), [`Data Aggregator`](data_aggregator.md), [`Detection Offset`](detection_offset.md), [`Keypoint Visualization`](keypoint_visualization.md), [`S3 Sink`](s3_sink.md), [`Camera Focus`](camera_focus.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Current Time`](current_time.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Ellipse Visualization`](ellipse_visualization.md), [`SAM 3`](sam3.md), [`Dynamic Zone`](dynamic_zone.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Cosine Similarity`](cosine_similarity.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Delta Filter`](delta_filter.md), [`Pixelate Visualization`](pixelate_visualization.md), [`OpenAI`](open_ai.md), [`Camera Focus`](camera_focus.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Property Definition`](property_definition.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Corner Visualization`](corner_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`LMM For Classification`](lmm_for_classification.md), [`PLC Reader`](plc_reader.md), [`Moondream2`](moondream2.md), [`Motion Detection`](motion_detection.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`CSV Formatter`](csv_formatter.md), [`SmolVLM2`](smol_vlm2.md), [`Qwen3.5`](qwen3.5.md), [`Contrast Equalization`](contrast_equalization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`EasyOCR`](easy_ocr.md), [`Template Matching`](template_matching.md), [`GLM-OCR`](glmocr.md), [`Cache Get`](cache_get.md), [`Circle Visualization`](circle_visualization.md), [`OpenAI`](open_ai.md), [`Barcode Detection`](barcode_detection.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Continue If`](continue_if.md), [`Relative Static Crop`](relative_static_crop.md), [`Gaze Detection`](gaze_detection.md), [`Path Deviation`](path_deviation.md), [`VLM As Detector`](vlm_as_detector.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Perspective Correction`](perspective_correction.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Trace Visualization`](trace_visualization.md), [`Overlap Analysis`](overlap_analysis.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Dominant Color`](dominant_color.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Florence-2 Model`](florence2_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Background Color Visualization`](background_color_visualization.md), [`Image Contours`](image_contours.md), [`Qwen-VL`](qwen_vl.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Clip Comparison`](clip_comparison.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Detections Consensus`](detections_consensus.md), [`Inner Workflow`](inner_workflow.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`JSON Parser`](json_parser.md), [`Image Threshold`](image_threshold.md), [`Image Blur`](image_blur.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Email Notification`](email_notification.md), [`Crop Visualization`](crop_visualization.md), [`Label Visualization`](label_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI`](open_ai.md), [`Buffer`](buffer.md), [`Distance Measurement`](distance_measurement.md), [`Triangle Visualization`](triangle_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Identify Changes`](identify_changes.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Halo Visualization`](halo_visualization.md), [`Switch Case`](switch_case.md), [`Expression`](expression.md), [`Detections Combine`](detections_combine.md), [`Byte Tracker`](byte_tracker.md), [`Overlap Filter`](overlap_filter.md), [`Grid Visualization`](grid_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Google Gemma`](google_gemma.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Detections Merge`](detections_merge.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Color Visualization`](color_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Webhook Sink`](webhook_sink.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Icon Visualization`](icon_visualization.md), [`OpenAI`](open_ai.md), [`Detections Filter`](detections_filter.md), [`SAM2 Video Tracker`](sam2_video_tracker.md)

    
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


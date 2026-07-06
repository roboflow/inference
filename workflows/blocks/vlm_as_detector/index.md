
# VLM As Detector



## v2

??? "Class: `VLMAsDetectorBlockV2`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/formatters/vlm_as_detector/v2.py">inference.core.workflows.core_steps.formatters.vlm_as_detector.v2.VLMAsDetectorBlockV2</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Parse JSON strings from Visual Language Models (VLMs) and Large Language Models (LLMs) into standardized object detection prediction format by extracting bounding boxes, class names, and confidences, converting normalized coordinates to pixel coordinates, mapping class names to class IDs, and handling multiple model types and task formats to enable VLM-based object detection, LLM detection parsing, and text-to-detection conversion workflows.

## How This Block Works

This block converts VLM/LLM text outputs containing object detection predictions into standardized object detection format compatible with workflow detection blocks. The block:

1. Receives image and VLM output string containing detection results in JSON format
2. Parses JSON content from VLM output:

   **Handles Markdown-wrapped JSON:**
   - Searches for JSON wrapped in Markdown code blocks (```json ... ```)
   - This format is common in LLM/VLM responses
   - If multiple markdown JSON blocks are found, only the first block is parsed
   - Extracts JSON content from within markdown tags

   **Handles raw JSON strings:**
   - If no markdown blocks are found, attempts to parse the entire string as JSON
   - Supports standard JSON format strings
3. Selects appropriate parser based on model type and task type:
   - Uses registered parsers that handle different model outputs (google-gemini, anthropic-claude, florence-2, openai)
   - Supports multiple task types: object-detection, open-vocabulary-object-detection, object-detection-and-caption, phrase-grounded-object-detection, region-proposal, ocr-with-text-detection
   - Each model/task combination uses a specialized parser for that format
4. Parses detection data based on model type:

   **For OpenAI/Gemini/Claude models:**
   - Extracts detections array from parsed JSON
   - Converts normalized coordinates (0-1 range) to pixel coordinates using image dimensions
   - Extracts class names, confidence scores, and bounding box coordinates
   - Maps class names to class IDs using provided classes list
   - Creates detection objects with bounding boxes, classes, and confidences

   **For Florence-2 model:**
   - Uses supervision's built-in LMM parser for Florence-2 format
   - Handles different task types with specialized parsing (object detection, open vocabulary, region proposal, OCR, etc.)
   - For region proposal tasks: assigns "roi" as class name
   - For open vocabulary detection: uses provided classes list for class ID mapping
   - For other tasks: uses MD5-based class ID generation or provided classes
   - Sets confidence to 1.0 for Florence-2 detections (model doesn't provide confidence)
5. Converts coordinates and normalizes data:
   - Converts normalized coordinates (0-1) to absolute pixel coordinates (x_min, y_min, x_max, y_max)
   - Scales coordinates using image width and height
   - Normalizes confidence scores to valid range [0.0, 1.0]
   - Clamps confidence values outside the range
6. Creates class name to class ID mapping:
   - For OpenAI/Gemini/Claude: uses provided classes list to create index mapping (class_name → class_id)
   - Classes are mapped in order (first class = ID 0, second = ID 1, etc.)
   - Classes not in the provided list get class_id = -1
   - For Florence-2: uses different mapping strategies based on task type
7. Constructs object detection predictions:
   - Creates supervision Detections objects with bounding boxes (xyxy format)
   - Includes class IDs, class names, and confidence scores
   - Adds metadata: detection IDs, inference IDs, image dimensions, prediction type
   - Attaches parent coordinates for crop-aware detections
   - Formats predictions in standard object detection format
8. Handles errors:
   - Sets `error_status` to True if JSON parsing fails
   - Sets `error_status` to True if detection parsing fails
   - Returns None for predictions when errors occur
   - Always includes inference_id for tracking
9. Returns object detection predictions:
   - Outputs `predictions` in standard object detection format (compatible with detection blocks)
   - Outputs `error_status` indicating parsing success/failure
   - Outputs `inference_id` for tracking and lineage

The block enables using VLMs/LLMs for object detection by converting their text-based JSON outputs into standardized detection predictions that can be used in workflows like any other object detection model output.

## Common Use Cases

- **VLM-Based Object Detection**: Use Visual Language Models for object detection by parsing VLM outputs into detection predictions (e.g., detect objects with GPT-4V, use Claude Vision for detection, parse Gemini detection outputs), enabling VLM detection workflows
- **Open-Vocabulary Detection**: Use VLMs for open-vocabulary object detection with custom classes (e.g., detect custom objects with VLMs, use open-vocabulary detection, detect objects not in training set), enabling open-vocabulary detection workflows
- **Multi-Task Detection**: Use VLMs for various detection tasks (e.g., object detection with captions, phrase-grounded detection, region proposal, OCR with detection), enabling multi-task detection workflows
- **LLM Detection Parsing**: Parse LLM text outputs containing detection results into standardized format (e.g., parse GPT detection outputs, convert LLM predictions to detection format, use LLMs for detection), enabling LLM detection workflows
- **Text-to-Detection Conversion**: Convert text-based detection outputs from models into workflow-compatible detection predictions (e.g., convert text predictions to detection format, parse text-based detections, convert model outputs to detections), enabling text-to-detection workflows
- **VLM Integration**: Integrate VLM outputs into detection workflows (e.g., use VLMs in detection pipelines, integrate VLM predictions with detection blocks, combine VLM and traditional detection), enabling VLM integration workflows

## Connecting to Other Blocks

This block receives images and VLM outputs and produces object detection predictions:

- **After VLM/LLM blocks** to parse detection outputs into standard format (e.g., VLM output to detections, LLM output to detections, parse model outputs), enabling VLM-to-detection workflows
- **Before detection-based blocks** to use parsed detections (e.g., use parsed detections in workflows, provide detections to downstream blocks, use VLM detections with detection blocks), enabling detection-to-workflow workflows
- **Before filtering blocks** to filter VLM detections (e.g., filter by class, filter by confidence, apply filters to VLM predictions), enabling detection-to-filter workflows
- **Before analytics blocks** to analyze VLM detection results (e.g., analyze VLM detections, perform analytics on parsed detections, track VLM detection metrics), enabling detection analytics workflows
- **Before visualization blocks** to display VLM detection results (e.g., visualize VLM detections, display parsed detection predictions, show VLM detection outputs), enabling detection visualization workflows
- **In workflow outputs** to provide VLM detections as final output (e.g., VLM detection outputs, parsed detection results, VLM-based detection outputs), enabling detection output workflows

## Version Differences

This version (v2) includes the following enhancements over v1:

- **Improved Type System**: The `inference_id` output now uses `INFERENCE_ID_KIND` instead of `STRING_KIND`, providing better type safety and semantic meaning for inference tracking identifiers in the workflow system
- **OpenAI Model Support**: Added support for OpenAI models in addition to Google Gemini, Anthropic Claude, and Florence-2 models, expanding the range of VLM/LLM models that can be used for object detection
- **Enhanced Type Safety**: Improved type system ensures better integration with workflow execution engine and provides clearer semantic meaning for inference tracking

## Requirements

This block requires an image input (for metadata and dimensions) and a VLM output string containing JSON detection data. The JSON can be raw JSON or wrapped in Markdown code blocks (```json ... ```). The block supports four model types: "openai", "google-gemini", "anthropic-claude", and "florence-2". It supports multiple task types: "object-detection", "open-vocabulary-object-detection", "object-detection-and-caption", "phrase-grounded-object-detection", "region-proposal", and "ocr-with-text-detection". The `classes` parameter is required for OpenAI, Gemini, and Claude models (to map class names to IDs) but optional for Florence-2 (some tasks don't require it). Classes are mapped to IDs by index (first class = 0, second = 1, etc.). Classes not in the list get class_id = -1. The block outputs object detection predictions in standard format (compatible with detection blocks), error_status (boolean), and inference_id (INFERENCE_ID_KIND) for tracking.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/vlm_as_detector@v2`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `classes` | `List[str]` | List of all class names used by the classification model, in order. Required to generate mapping between class names (from VLM output) and class IDs (for detection format). Classes are mapped to IDs by index: first class = ID 0, second = ID 1, etc. Classes from VLM output that are not in this list get class_id = -1. Required for OpenAI, Gemini, and Claude models. Optional for Florence-2 (some tasks don't require it). Should match the classes the VLM was asked to detect.. | ✅ |
| `model_type` | `str` | Type of the VLM/LLM model that generated the prediction. Determines which parser is used to extract detection data from the JSON output. Supported models: 'openai' (GPT-4V), 'google-gemini' (Gemini Vision), 'anthropic-claude' (Claude Vision), 'florence-2' (Microsoft Florence-2). Each model type has different JSON output formats, so the correct model type must be specified for proper parsing.. | ❌ |
| `task_type` | `str` | Task type performed by the VLM/LLM model. Determines which parser and format handler is used. Supported task types: 'object-detection' (standard object detection), 'open-vocabulary-object-detection' (detect objects with custom classes), 'object-detection-and-caption' (detection with captions), 'phrase-grounded-object-detection' (ground phrases to detections), 'region-proposal' (propose regions of interest), 'ocr-with-text-detection' (OCR with text region detection). The task type must match what the VLM/LLM was asked to perform.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `VLM As Detector` in version `v2`.

    - inputs: [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Google Gemma API`](google_gemma_api.md), [`Qwen-VL`](qwen_vl.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Image Stack`](image_stack.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Triangle Visualization`](triangle_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Dynamic Zone`](dynamic_zone.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Camera Focus`](camera_focus.md), [`Image Slicer`](image_slicer.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Stitch Images`](stitch_images.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Polygon Visualization`](polygon_visualization.md), [`Florence-2 Model`](florence2_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dynamic Crop`](dynamic_crop.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Perspective Correction`](perspective_correction.md), [`Google Gemini`](google_gemini.md), [`Polygon Visualization`](polygon_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`GLM-OCR`](glmocr.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Clip Comparison`](clip_comparison.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Circle Visualization`](circle_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Grid Visualization`](grid_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Buffer`](buffer.md), [`Image Blur`](image_blur.md), [`Label Visualization`](label_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Anthropic Claude`](anthropic_claude.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Image Preprocessing`](image_preprocessing.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Trace Visualization`](trace_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Color Visualization`](color_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Dimension Collapse`](dimension_collapse.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Calibration`](camera_calibration.md), [`Dot Visualization`](dot_visualization.md), [`Motion Detection`](motion_detection.md), [`Halo Visualization`](halo_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Image Contours`](image_contours.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`Halo Visualization`](halo_visualization.md), [`OpenRouter`](open_router.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Gemma`](google_gemma.md), [`Relative Static Crop`](relative_static_crop.md), [`Contrast Equalization`](contrast_equalization.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Google Gemini`](google_gemini.md), [`Text Display`](text_display.md), [`Size Measurement`](size_measurement.md), [`Depth Estimation`](depth_estimation.md), [`Image Threshold`](image_threshold.md), [`Icon Visualization`](icon_visualization.md), [`Blur Visualization`](blur_visualization.md), [`OpenAI`](open_ai.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Google Gemini`](google_gemini.md), [`GeoTag Detection`](geo_tag_detection.md), [`SIFT`](sift.md), [`Background Subtraction`](background_subtraction.md), [`Background Color Visualization`](background_color_visualization.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Mask Visualization`](mask_visualization.md)
    - outputs: [`Twilio SMS Notification`](twilio_sms_notification.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`SAM 3`](sam3.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Path Deviation`](path_deviation.md), [`Detections Filter`](detections_filter.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Image Stack`](image_stack.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Gaze Detection`](gaze_detection.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Triangle Visualization`](triangle_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Dynamic Zone`](dynamic_zone.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Overlap Filter`](overlap_filter.md), [`Track Class Lock`](track_class_lock.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Time in Zone`](timein_zone.md), [`Polygon Visualization`](polygon_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Dynamic Crop`](dynamic_crop.md), [`Event Writer`](event_writer.md), [`Perspective Correction`](perspective_correction.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Line Counter`](line_counter.md), [`Polygon Visualization`](polygon_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Circle Visualization`](circle_visualization.md), [`Slack Notification`](slack_notification.md), [`Line Counter`](line_counter.md), [`Distance Measurement`](distance_measurement.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Time in Zone`](timein_zone.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Corner Visualization`](corner_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Florence-2 Model`](florence2_model.md), [`Template Matching`](template_matching.md), [`Label Visualization`](label_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Overlap Analysis`](overlap_analysis.md), [`SORT Tracker`](sort_tracker.md), [`Object Detection Model`](object_detection_model.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Path Deviation`](path_deviation.md), [`Velocity`](velocity.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Trace Visualization`](trace_visualization.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Email Notification`](email_notification.md), [`SIFT Comparison`](sift_comparison.md), [`Detections Consensus`](detections_consensus.md), [`Camera Calibration`](camera_calibration.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dot Visualization`](dot_visualization.md), [`Motion Detection`](motion_detection.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Halo Visualization`](halo_visualization.md), [`Detection Offset`](detection_offset.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Camera Focus`](camera_focus.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Halo Visualization`](halo_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Detections Combine`](detections_combine.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Detections Merge`](detections_merge.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Google Gemini`](google_gemini.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Email Notification`](email_notification.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Byte Tracker`](byte_tracker.md), [`SAM 3`](sam3.md), [`MQTT Writer`](mqtt_writer.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Detections Stitch`](detections_stitch.md), [`Text Display`](text_display.md), [`Size Measurement`](size_measurement.md), [`Icon Visualization`](icon_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Time in Zone`](timein_zone.md), [`GeoTag Detection`](geo_tag_detection.md), [`PLC Writer`](plc_writer.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Background Color Visualization`](background_color_visualization.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Mask Visualization`](mask_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`VLM As Detector` in version `v2`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Input image that was used to generate the VLM prediction. Used to extract image dimensions (width, height) for converting normalized coordinates to pixel coordinates and metadata (parent_id) for the detection predictions. The same image that was provided to the VLM/LLM block should be used here to maintain consistency..
        - `vlm_output` (*[`language_model_output`](../kinds/language_model_output.md)*): String output from a VLM or LLM block containing object detection prediction in JSON format. Can be raw JSON string or JSON wrapped in Markdown code blocks (e.g., ```json {...} ```). Format depends on model_type and task_type - different models and tasks produce different JSON structures. If multiple markdown blocks exist, only the first is parsed..
        - `classes` (*[`list_of_values`](../kinds/list_of_values.md)*): List of all class names used by the classification model, in order. Required to generate mapping between class names (from VLM output) and class IDs (for detection format). Classes are mapped to IDs by index: first class = ID 0, second = ID 1, etc. Classes from VLM output that are not in this list get class_id = -1. Required for OpenAI, Gemini, and Claude models. Optional for Florence-2 (some tasks don't require it). Should match the classes the VLM was asked to detect..

    - output
    
        - `error_status` ([`boolean`](../kinds/boolean.md)): Boolean flag.
        - `predictions` ([`object_detection_prediction`](../kinds/object_detection_prediction.md)): Prediction with detected bounding boxes in form of sv.Detections(...) object.
        - `inference_id` ([`inference_id`](../kinds/inference_id.md)): Inference identifier.



??? tip "Example JSON definition of step `VLM As Detector` in version `v2`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/vlm_as_detector@v2",
	    "image": "$inputs.image",
	    "vlm_output": [
	        "$steps.lmm.output"
	    ],
	    "classes": [
	        "$steps.lmm.classes",
	        "$inputs.classes",
	        [
	            "dog",
	            "cat",
	            "bird"
	        ],
	        [
	            "class_a",
	            "class_b"
	        ]
	    ],
	    "model_type": [
	        "openai"
	    ],
	    "task_type": "<block_does_not_provide_example>"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    




## v1

??? "Class: `VLMAsDetectorBlockV1`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/formatters/vlm_as_detector/v1.py">inference.core.workflows.core_steps.formatters.vlm_as_detector.v1.VLMAsDetectorBlockV1</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Parse JSON strings from Visual Language Models (VLMs) and Large Language Models (LLMs) into standardized object detection prediction format by extracting bounding boxes, class names, and confidences, converting normalized coordinates to pixel coordinates, mapping class names to class IDs, and handling multiple model types and task formats to enable VLM-based object detection, LLM detection parsing, and text-to-detection conversion workflows.

## How This Block Works

This block converts VLM/LLM text outputs containing object detection predictions into standardized object detection format compatible with workflow detection blocks. The block:

1. Receives image and VLM output string containing detection results in JSON format
2. Parses JSON content from VLM output:

   **Handles Markdown-wrapped JSON:**
   - Searches for JSON wrapped in Markdown code blocks (```json ... ```)
   - This format is common in LLM/VLM responses
   - If multiple markdown JSON blocks are found, only the first block is parsed
   - Extracts JSON content from within markdown tags

   **Handles raw JSON strings:**
   - If no markdown blocks are found, attempts to parse the entire string as JSON
   - Supports standard JSON format strings
3. Selects appropriate parser based on model type and task type:
   - Uses registered parsers that handle different model outputs (google-gemini, anthropic-claude, florence-2)
   - Supports multiple task types: object-detection, open-vocabulary-object-detection, object-detection-and-caption, phrase-grounded-object-detection, region-proposal, ocr-with-text-detection
   - Each model/task combination uses a specialized parser for that format
4. Parses detection data based on model type:

   **For Gemini/Claude models:**
   - Extracts detections array from parsed JSON
   - Converts normalized coordinates (0-1 range) to pixel coordinates using image dimensions
   - Extracts class names, confidence scores, and bounding box coordinates
   - Maps class names to class IDs using provided classes list
   - Creates detection objects with bounding boxes, classes, and confidences

   **For Florence-2 model:**
   - Uses supervision's built-in LMM parser for Florence-2 format
   - Handles different task types with specialized parsing (object detection, open vocabulary, region proposal, OCR, etc.)
   - For region proposal tasks: assigns "roi" as class name
   - For open vocabulary detection: uses provided classes list for class ID mapping
   - For other tasks: uses MD5-based class ID generation or provided classes
   - Sets confidence to 1.0 for Florence-2 detections (model doesn't provide confidence)
5. Converts coordinates and normalizes data:
   - Converts normalized coordinates (0-1) to absolute pixel coordinates (x_min, y_min, x_max, y_max)
   - Scales coordinates using image width and height
   - Normalizes confidence scores to valid range [0.0, 1.0]
   - Clamps confidence values outside the range
6. Creates class name to class ID mapping:
   - For Gemini/Claude: uses provided classes list to create index mapping (class_name → class_id)
   - Classes are mapped in order (first class = ID 0, second = ID 1, etc.)
   - Classes not in the provided list get class_id = -1
   - For Florence-2: uses different mapping strategies based on task type
7. Constructs object detection predictions:
   - Creates supervision Detections objects with bounding boxes (xyxy format)
   - Includes class IDs, class names, and confidence scores
   - Adds metadata: detection IDs, inference IDs, image dimensions, prediction type
   - Attaches parent coordinates for crop-aware detections
   - Formats predictions in standard object detection format
8. Handles errors:
   - Sets `error_status` to True if JSON parsing fails
   - Sets `error_status` to True if detection parsing fails
   - Returns None for predictions when errors occur
   - Always includes inference_id for tracking
9. Returns object detection predictions:
   - Outputs `predictions` in standard object detection format (compatible with detection blocks)
   - Outputs `error_status` indicating parsing success/failure
   - Outputs `inference_id` for tracking and lineage

The block enables using VLMs/LLMs for object detection by converting their text-based JSON outputs into standardized detection predictions that can be used in workflows like any other object detection model output.

## Common Use Cases

- **VLM-Based Object Detection**: Use Visual Language Models for object detection by parsing VLM outputs into detection predictions (e.g., detect objects with GPT-4V, use Claude Vision for detection, parse Gemini detection outputs), enabling VLM detection workflows
- **Open-Vocabulary Detection**: Use VLMs for open-vocabulary object detection with custom classes (e.g., detect custom objects with VLMs, use open-vocabulary detection, detect objects not in training set), enabling open-vocabulary detection workflows
- **Multi-Task Detection**: Use VLMs for various detection tasks (e.g., object detection with captions, phrase-grounded detection, region proposal, OCR with detection), enabling multi-task detection workflows
- **LLM Detection Parsing**: Parse LLM text outputs containing detection results into standardized format (e.g., parse GPT detection outputs, convert LLM predictions to detection format, use LLMs for detection), enabling LLM detection workflows
- **Text-to-Detection Conversion**: Convert text-based detection outputs from models into workflow-compatible detection predictions (e.g., convert text predictions to detection format, parse text-based detections, convert model outputs to detections), enabling text-to-detection workflows
- **VLM Integration**: Integrate VLM outputs into detection workflows (e.g., use VLMs in detection pipelines, integrate VLM predictions with detection blocks, combine VLM and traditional detection), enabling VLM integration workflows

## Connecting to Other Blocks

This block receives images and VLM outputs and produces object detection predictions:

- **After VLM/LLM blocks** to parse detection outputs into standard format (e.g., VLM output to detections, LLM output to detections, parse model outputs), enabling VLM-to-detection workflows
- **Before detection-based blocks** to use parsed detections (e.g., use parsed detections in workflows, provide detections to downstream blocks, use VLM detections with detection blocks), enabling detection-to-workflow workflows
- **Before filtering blocks** to filter VLM detections (e.g., filter by class, filter by confidence, apply filters to VLM predictions), enabling detection-to-filter workflows
- **Before analytics blocks** to analyze VLM detection results (e.g., analyze VLM detections, perform analytics on parsed detections, track VLM detection metrics), enabling detection analytics workflows
- **Before visualization blocks** to display VLM detection results (e.g., visualize VLM detections, display parsed detection predictions, show VLM detection outputs), enabling detection visualization workflows
- **In workflow outputs** to provide VLM detections as final output (e.g., VLM detection outputs, parsed detection results, VLM-based detection outputs), enabling detection output workflows

## Requirements

This block requires an image input (for metadata and dimensions) and a VLM output string containing JSON detection data. The JSON can be raw JSON or wrapped in Markdown code blocks (```json ... ```). The block supports three model types: "google-gemini", "anthropic-claude", and "florence-2". It supports multiple task types: "object-detection", "open-vocabulary-object-detection", "object-detection-and-caption", "phrase-grounded-object-detection", "region-proposal", and "ocr-with-text-detection". The `classes` parameter is required for Gemini and Claude models (to map class names to IDs) but optional for Florence-2 (some tasks don't require it). Classes are mapped to IDs by index (first class = 0, second = 1, etc.). Classes not in the list get class_id = -1. The block outputs object detection predictions in standard format (compatible with detection blocks), error_status (boolean), and inference_id (string) for tracking.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/vlm_as_detector@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `classes` | `List[str]` | List of all class names used by the detection model, in order. Required for google-gemini and anthropic-claude models to generate mapping between class names (from VLM output) and class IDs (for detection format). Optional for florence-2 model (required only for open-vocabulary-object-detection task). Classes are mapped to IDs by index: first class = ID 0, second = ID 1, etc. Classes from VLM output that are not in this list get class_id = -1. Should match the classes the VLM was asked to detect.. | ✅ |
| `model_type` | `str` | Type of VLM/LLM model that generated the detection prediction. Determines which parser to use for parsing the JSON output. 'google-gemini': Google Gemini model outputs. 'anthropic-claude': Anthropic Claude model outputs. 'florence-2': Microsoft Florence-2 model outputs. Each model type has different JSON output formats and requires appropriate parsing.. | ❌ |
| `task_type` | `str` | Task type that was performed by the VLM model. Determines how the JSON output is parsed and what detection format is expected. Supported tasks: 'object-detection' (unprompted detection), 'open-vocabulary-object-detection' (detection with provided classes), 'object-detection-and-caption' (detection with captions), 'phrase-grounded-object-detection' (prompted detection), 'region-proposal' (regions of interest), 'ocr-with-text-detection' (text detection with OCR). Each task type has specific output format requirements.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `VLM As Detector` in version `v1`.

    - inputs: [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Google Gemma API`](google_gemma_api.md), [`Qwen-VL`](qwen_vl.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Image Stack`](image_stack.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Triangle Visualization`](triangle_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Dynamic Zone`](dynamic_zone.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Camera Focus`](camera_focus.md), [`Image Slicer`](image_slicer.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Stitch Images`](stitch_images.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Polygon Visualization`](polygon_visualization.md), [`Florence-2 Model`](florence2_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dynamic Crop`](dynamic_crop.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Perspective Correction`](perspective_correction.md), [`Google Gemini`](google_gemini.md), [`Polygon Visualization`](polygon_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`GLM-OCR`](glmocr.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Clip Comparison`](clip_comparison.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Circle Visualization`](circle_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Grid Visualization`](grid_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Buffer`](buffer.md), [`Image Blur`](image_blur.md), [`Label Visualization`](label_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Anthropic Claude`](anthropic_claude.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Image Preprocessing`](image_preprocessing.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Trace Visualization`](trace_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Color Visualization`](color_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Dimension Collapse`](dimension_collapse.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Calibration`](camera_calibration.md), [`Dot Visualization`](dot_visualization.md), [`Motion Detection`](motion_detection.md), [`Halo Visualization`](halo_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Image Contours`](image_contours.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`Halo Visualization`](halo_visualization.md), [`OpenRouter`](open_router.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Gemma`](google_gemma.md), [`Relative Static Crop`](relative_static_crop.md), [`Contrast Equalization`](contrast_equalization.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Google Gemini`](google_gemini.md), [`Text Display`](text_display.md), [`Size Measurement`](size_measurement.md), [`Depth Estimation`](depth_estimation.md), [`Image Threshold`](image_threshold.md), [`Icon Visualization`](icon_visualization.md), [`Blur Visualization`](blur_visualization.md), [`OpenAI`](open_ai.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Google Gemini`](google_gemini.md), [`GeoTag Detection`](geo_tag_detection.md), [`SIFT`](sift.md), [`Background Subtraction`](background_subtraction.md), [`Background Color Visualization`](background_color_visualization.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Mask Visualization`](mask_visualization.md)
    - outputs: [`Detections Filter`](detections_filter.md), [`Qwen-VL`](qwen_vl.md), [`Gaze Detection`](gaze_detection.md), [`Cache Get`](cache_get.md), [`Triangle Visualization`](triangle_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Track Class Lock`](track_class_lock.md), [`YOLO-World Model`](yolo_world_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Google Gemini`](google_gemini.md), [`GLM-OCR`](glmocr.md), [`QR Code Generator`](qr_code_generator.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Line Counter`](line_counter.md), [`Distance Measurement`](distance_measurement.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Corner Visualization`](corner_visualization.md), [`Overlap Analysis`](overlap_analysis.md), [`Template Matching`](template_matching.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Velocity`](velocity.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`LMM`](lmm.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Morphological Transformation`](morphological_transformation.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Calibration`](camera_calibration.md), [`Motion Detection`](motion_detection.md), [`Detection Offset`](detection_offset.md), [`Google Vision OCR`](google_vision_ocr.md), [`Clip Comparison`](clip_comparison.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`OpenAI`](open_ai.md), [`Halo Visualization`](halo_visualization.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Detections Combine`](detections_combine.md), [`Contrast Equalization`](contrast_equalization.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`MQTT Writer`](mqtt_writer.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Cache Set`](cache_set.md), [`Image Threshold`](image_threshold.md), [`Blur Visualization`](blur_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Google Gemini`](google_gemini.md), [`Background Color Visualization`](background_color_visualization.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Object Detection Model`](object_detection_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`SAM 3`](sam3.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Path Deviation`](path_deviation.md), [`Google Gemma API`](google_gemma_api.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Image Stack`](image_stack.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Crop Visualization`](crop_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Dynamic Zone`](dynamic_zone.md), [`Overlap Filter`](overlap_filter.md), [`Florence-2 Model`](florence2_model.md), [`Time in Zone`](timein_zone.md), [`Polygon Visualization`](polygon_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dynamic Crop`](dynamic_crop.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Event Writer`](event_writer.md), [`Perspective Correction`](perspective_correction.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Line Counter`](line_counter.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Circle Visualization`](circle_visualization.md), [`Slack Notification`](slack_notification.md), [`CogVLM`](cog_vlm.md), [`Time in Zone`](timein_zone.md), [`Byte Tracker`](byte_tracker.md), [`LMM For Classification`](lmm_for_classification.md), [`Florence-2 Model`](florence2_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`OpenAI`](open_ai.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`SORT Tracker`](sort_tracker.md), [`Local File Sink`](local_file_sink.md), [`Label Visualization`](label_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Image Blur`](image_blur.md), [`Seg Preview`](seg_preview.md), [`Object Detection Model`](object_detection_model.md), [`Path Deviation`](path_deviation.md), [`Anthropic Claude`](anthropic_claude.md), [`Pixel Color Count`](pixel_color_count.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Preprocessing`](image_preprocessing.md), [`Trace Visualization`](trace_visualization.md), [`Email Notification`](email_notification.md), [`SIFT Comparison`](sift_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Detections Consensus`](detections_consensus.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dot Visualization`](dot_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Current Time`](current_time.md), [`Halo Visualization`](halo_visualization.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Camera Focus`](camera_focus.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenRouter`](open_router.md), [`Moondream2`](moondream2.md), [`S3 Sink`](s3_sink.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Gemma`](google_gemma.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`OpenAI`](open_ai.md), [`Detections Merge`](detections_merge.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemini`](google_gemini.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Email Notification`](email_notification.md), [`Byte Tracker`](byte_tracker.md), [`SAM 3`](sam3.md), [`Detections Stitch`](detections_stitch.md), [`Text Display`](text_display.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Size Measurement`](size_measurement.md), [`Depth Estimation`](depth_estimation.md), [`Icon Visualization`](icon_visualization.md), [`OpenAI`](open_ai.md), [`Time in Zone`](timein_zone.md), [`GeoTag Detection`](geo_tag_detection.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`PLC Writer`](plc_writer.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Mask Visualization`](mask_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`VLM As Detector` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Input image that was used to generate the VLM prediction. Used to extract image dimensions (width, height) for converting normalized coordinates to pixel coordinates and metadata (parent_id) for the detection predictions. The same image that was provided to the VLM/LLM block should be used here to maintain consistency..
        - `vlm_output` (*[`language_model_output`](../kinds/language_model_output.md)*): String output from a VLM or LLM block containing object detection prediction in JSON format. Can be raw JSON string or JSON wrapped in Markdown code blocks (e.g., ```json {...} ```). Format depends on model_type and task_type - different models and tasks produce different JSON structures. If multiple markdown blocks exist, only the first is parsed..
        - `classes` (*[`list_of_values`](../kinds/list_of_values.md)*): List of all class names used by the detection model, in order. Required for google-gemini and anthropic-claude models to generate mapping between class names (from VLM output) and class IDs (for detection format). Optional for florence-2 model (required only for open-vocabulary-object-detection task). Classes are mapped to IDs by index: first class = ID 0, second = ID 1, etc. Classes from VLM output that are not in this list get class_id = -1. Should match the classes the VLM was asked to detect..

    - output
    
        - `error_status` ([`boolean`](../kinds/boolean.md)): Boolean flag.
        - `predictions` ([`object_detection_prediction`](../kinds/object_detection_prediction.md)): Prediction with detected bounding boxes in form of sv.Detections(...) object.
        - `inference_id` ([`string`](../kinds/string.md)): String value.



??? tip "Example JSON definition of step `VLM As Detector` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/vlm_as_detector@v1",
	    "image": "$inputs.image",
	    "vlm_output": [
	        "$steps.lmm.output"
	    ],
	    "classes": [
	        "$steps.lmm.classes",
	        "$inputs.classes",
	        [
	            "dog",
	            "cat",
	            "bird"
	        ],
	        [
	            "class_a",
	            "class_b"
	        ]
	    ],
	    "model_type": "google-gemini",
	    "task_type": "<block_does_not_provide_example>"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    


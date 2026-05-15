
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

    - inputs: [`Perspective Correction`](perspective_correction.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Clip Comparison`](clip_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Qwen-VL`](qwen_vl.md), [`QR Code Generator`](qr_code_generator.md), [`OpenRouter`](open_router.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Image Threshold`](image_threshold.md), [`Anthropic Claude`](anthropic_claude.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`OpenAI`](open_ai.md), [`Dynamic Crop`](dynamic_crop.md), [`Size Measurement`](size_measurement.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Anthropic Claude`](anthropic_claude.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Clip Comparison`](clip_comparison.md), [`Camera Focus`](camera_focus.md), [`Label Visualization`](label_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Depth Estimation`](depth_estimation.md), [`Google Gemini`](google_gemini.md), [`Image Contours`](image_contours.md), [`Relative Static Crop`](relative_static_crop.md), [`Motion Detection`](motion_detection.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Background Color Visualization`](background_color_visualization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Image Blur`](image_blur.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemini`](google_gemini.md), [`SIFT Comparison`](sift_comparison.md), [`Grid Visualization`](grid_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Florence-2 Model`](florence2_model.md), [`Triangle Visualization`](triangle_visualization.md), [`OpenAI`](open_ai.md), [`Image Stack`](image_stack.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Stitch Images`](stitch_images.md), [`Buffer`](buffer.md), [`Image Slicer`](image_slicer.md), [`Image Preprocessing`](image_preprocessing.md), [`SIFT`](sift.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Slicer`](image_slicer.md), [`Dynamic Zone`](dynamic_zone.md), [`Corner Visualization`](corner_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Halo Visualization`](halo_visualization.md), [`Color Visualization`](color_visualization.md), [`Google Gemini`](google_gemini.md), [`Blur Visualization`](blur_visualization.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Camera Focus`](camera_focus.md), [`Camera Calibration`](camera_calibration.md), [`Morphological Transformation`](morphological_transformation.md), [`Trace Visualization`](trace_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Mask Visualization`](mask_visualization.md), [`GLM-OCR`](glmocr.md), [`Crop Visualization`](crop_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Circle Visualization`](circle_visualization.md), [`Text Display`](text_display.md), [`Dimension Collapse`](dimension_collapse.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Florence-2 Model`](florence2_model.md), [`Contrast Equalization`](contrast_equalization.md), [`Icon Visualization`](icon_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemma`](google_gemma.md)
    - outputs: [`Object Detection Model`](object_detection_model.md), [`Perspective Correction`](perspective_correction.md), [`SAM 3`](sam3.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Email Notification`](email_notification.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Path Deviation`](path_deviation.md), [`SAM 3`](sam3.md), [`Object Detection Model`](object_detection_model.md), [`Line Counter`](line_counter.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Dynamic Crop`](dynamic_crop.md), [`Detections Consensus`](detections_consensus.md), [`Size Measurement`](size_measurement.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Email Notification`](email_notification.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Label Visualization`](label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Path Deviation`](path_deviation.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Overlap Filter`](overlap_filter.md), [`Detection Offset`](detection_offset.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Motion Detection`](motion_detection.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Background Color Visualization`](background_color_visualization.md), [`Template Matching`](template_matching.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Velocity`](velocity.md), [`SIFT Comparison`](sift_comparison.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Detection Event Log`](detection_event_log.md), [`Florence-2 Model`](florence2_model.md), [`Object Detection Model`](object_detection_model.md), [`Triangle Visualization`](triangle_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Time in Zone`](timein_zone.md), [`Slack Notification`](slack_notification.md), [`Image Stack`](image_stack.md), [`Detections Filter`](detections_filter.md), [`Detections Merge`](detections_merge.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Dynamic Zone`](dynamic_zone.md), [`Corner Visualization`](corner_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Halo Visualization`](halo_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Detections Transformation`](detections_transformation.md), [`Time in Zone`](timein_zone.md), [`Google Gemini`](google_gemini.md), [`Color Visualization`](color_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Camera Focus`](camera_focus.md), [`Camera Calibration`](camera_calibration.md), [`Distance Measurement`](distance_measurement.md), [`Trace Visualization`](trace_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Gaze Detection`](gaze_detection.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Dot Visualization`](dot_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Mask Visualization`](mask_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Text Display`](text_display.md), [`Detections Combine`](detections_combine.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Florence-2 Model`](florence2_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Byte Tracker`](byte_tracker.md), [`Webhook Sink`](webhook_sink.md), [`Icon Visualization`](icon_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Mask Area Measurement`](mask_area_measurement.md)

    
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

    - inputs: [`Perspective Correction`](perspective_correction.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Clip Comparison`](clip_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Qwen-VL`](qwen_vl.md), [`QR Code Generator`](qr_code_generator.md), [`OpenRouter`](open_router.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Image Threshold`](image_threshold.md), [`Anthropic Claude`](anthropic_claude.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`OpenAI`](open_ai.md), [`Dynamic Crop`](dynamic_crop.md), [`Size Measurement`](size_measurement.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Anthropic Claude`](anthropic_claude.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Clip Comparison`](clip_comparison.md), [`Camera Focus`](camera_focus.md), [`Label Visualization`](label_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Depth Estimation`](depth_estimation.md), [`Google Gemini`](google_gemini.md), [`Image Contours`](image_contours.md), [`Relative Static Crop`](relative_static_crop.md), [`Motion Detection`](motion_detection.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Background Color Visualization`](background_color_visualization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Image Blur`](image_blur.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemini`](google_gemini.md), [`SIFT Comparison`](sift_comparison.md), [`Grid Visualization`](grid_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Florence-2 Model`](florence2_model.md), [`Triangle Visualization`](triangle_visualization.md), [`OpenAI`](open_ai.md), [`Image Stack`](image_stack.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Stitch Images`](stitch_images.md), [`Buffer`](buffer.md), [`Image Slicer`](image_slicer.md), [`Image Preprocessing`](image_preprocessing.md), [`SIFT`](sift.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Slicer`](image_slicer.md), [`Dynamic Zone`](dynamic_zone.md), [`Corner Visualization`](corner_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Halo Visualization`](halo_visualization.md), [`Color Visualization`](color_visualization.md), [`Google Gemini`](google_gemini.md), [`Blur Visualization`](blur_visualization.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Camera Focus`](camera_focus.md), [`Camera Calibration`](camera_calibration.md), [`Morphological Transformation`](morphological_transformation.md), [`Trace Visualization`](trace_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Mask Visualization`](mask_visualization.md), [`GLM-OCR`](glmocr.md), [`Crop Visualization`](crop_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Circle Visualization`](circle_visualization.md), [`Text Display`](text_display.md), [`Dimension Collapse`](dimension_collapse.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Florence-2 Model`](florence2_model.md), [`Contrast Equalization`](contrast_equalization.md), [`Icon Visualization`](icon_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemma`](google_gemma.md)
    - outputs: [`S3 Sink`](s3_sink.md), [`Email Notification`](email_notification.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Path Deviation`](path_deviation.md), [`SAM 3`](sam3.md), [`Morphological Transformation`](morphological_transformation.md), [`Qwen-VL`](qwen_vl.md), [`Clip Comparison`](clip_comparison.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`YOLO-World Model`](yolo_world_model.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`OpenAI`](open_ai.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Email Notification`](email_notification.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Anthropic Claude`](anthropic_claude.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Seg Preview`](seg_preview.md), [`Google Vision OCR`](google_vision_ocr.md), [`Label Visualization`](label_visualization.md), [`SAM 3`](sam3.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Path Deviation`](path_deviation.md), [`Overlap Filter`](overlap_filter.md), [`Local File Sink`](local_file_sink.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Google Gemini`](google_gemini.md), [`Motion Detection`](motion_detection.md), [`Byte Tracker`](byte_tracker.md), [`Background Color Visualization`](background_color_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Google Gemini`](google_gemini.md), [`Polygon Visualization`](polygon_visualization.md), [`Moondream2`](moondream2.md), [`Velocity`](velocity.md), [`SIFT Comparison`](sift_comparison.md), [`Detection Event Log`](detection_event_log.md), [`Florence-2 Model`](florence2_model.md), [`Time in Zone`](timein_zone.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Detections Filter`](detections_filter.md), [`Detections Merge`](detections_merge.md), [`Detections Stabilizer`](detections_stabilizer.md), [`LMM For Classification`](lmm_for_classification.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Preprocessing`](image_preprocessing.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Dynamic Zone`](dynamic_zone.md), [`Corner Visualization`](corner_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Halo Visualization`](halo_visualization.md), [`Time in Zone`](timein_zone.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Blur Visualization`](blur_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Distance Measurement`](distance_measurement.md), [`Morphological Transformation`](morphological_transformation.md), [`Trace Visualization`](trace_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Gaze Detection`](gaze_detection.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Text Display`](text_display.md), [`Detections Combine`](detections_combine.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Florence-2 Model`](florence2_model.md), [`Byte Tracker`](byte_tracker.md), [`Icon Visualization`](icon_visualization.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Object Detection Model`](object_detection_model.md), [`Perspective Correction`](perspective_correction.md), [`SAM 3`](sam3.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Object Detection Model`](object_detection_model.md), [`Line Counter`](line_counter.md), [`QR Code Generator`](qr_code_generator.md), [`OpenRouter`](open_router.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Image Threshold`](image_threshold.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Dynamic Crop`](dynamic_crop.md), [`Detections Consensus`](detections_consensus.md), [`Size Measurement`](size_measurement.md), [`Cache Set`](cache_set.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Detection Offset`](detection_offset.md), [`Depth Estimation`](depth_estimation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Template Matching`](template_matching.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Image Blur`](image_blur.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Anthropic Claude`](anthropic_claude.md), [`Object Detection Model`](object_detection_model.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`OpenAI`](open_ai.md), [`Slack Notification`](slack_notification.md), [`Image Stack`](image_stack.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Cache Get`](cache_get.md), [`LMM`](lmm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Detections Transformation`](detections_transformation.md), [`Color Visualization`](color_visualization.md), [`Google Gemini`](google_gemini.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Camera Focus`](camera_focus.md), [`Camera Calibration`](camera_calibration.md), [`Detections Stitch`](detections_stitch.md), [`Byte Tracker`](byte_tracker.md), [`Ellipse Visualization`](ellipse_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`SORT Tracker`](sort_tracker.md), [`Mask Visualization`](mask_visualization.md), [`GLM-OCR`](glmocr.md), [`Crop Visualization`](crop_visualization.md), [`Circle Visualization`](circle_visualization.md), [`CogVLM`](cog_vlm.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Contrast Equalization`](contrast_equalization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Webhook Sink`](webhook_sink.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemma`](google_gemma.md)

    
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


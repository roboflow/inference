
# VLM As Classifier



## v2

??? "Class: `VLMAsClassifierBlockV2`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/formatters/vlm_as_classifier/v2.py">inference.core.workflows.core_steps.formatters.vlm_as_classifier.v2.VLMAsClassifierBlockV2</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Parse JSON strings from Visual Language Models (VLMs) and Large Language Models (LLMs) into standardized classification prediction format by extracting class predictions, mapping class names to class IDs, handling both single-class and multi-label formats, and converting VLM/LLM text outputs into workflow-compatible classification results for VLM-based classification, LLM classification parsing, and text-to-classification conversion workflows.

## How This Block Works

This block converts VLM/LLM text outputs containing classification predictions into standardized classification prediction format. The block:

1. Receives image and VLM output string containing classification results in JSON format
2. Parses JSON content from VLM output:

   **Handles Markdown-wrapped JSON:**
   - Searches for JSON wrapped in Markdown code blocks (```json ... ```)
   - This format is common in LLM/VLM responses
   - If multiple markdown JSON blocks are found, only the first block is parsed
   - Extracts JSON content from within markdown tags

   **Handles raw JSON strings:**
   - If no markdown blocks are found, attempts to parse the entire string as JSON
   - Supports standard JSON format strings
3. Detects classification format and parses accordingly:

   **Single-Class Classification Format:**
   - Detects format containing "class_name" and "confidence" fields
   - Extracts the predicted class name and confidence score
   - Creates classification prediction with single top class
   - Maps class name to class ID using provided classes list

   **Multi-Label Classification Format:**
   - Detects format containing "predicted_classes" array
   - Extracts all predicted classes with their confidence scores
   - Handles duplicate classes by taking maximum confidence
   - Maps all class names to class IDs using provided classes list
4. Creates class name to class ID mapping:
   - Uses the provided classes list to create index mapping (class_name → class_id)
   - Maps classes in order (first class = ID 0, second = ID 1, etc.)
   - Classes not in the provided list get class_id = -1
5. Normalizes confidence scores:
   - Scales confidence values to valid range [0.0, 1.0]
   - Clamps values outside the range to 0.0 or 1.0
6. Constructs classification prediction:
   - Includes image dimensions (width, height) from input image
   - For single-class: includes "top" class, confidence, and predictions array
   - For multi-label: includes "predicted_classes" list and predictions dictionary
   - Includes inference_id and parent_id for tracking
   - Formats prediction in standard classification prediction format
7. Handles errors:
   - Sets `error_status` to True if JSON parsing fails
   - Sets `error_status` to True if classification format cannot be determined
   - Returns None for predictions when errors occur
   - Always includes inference_id for tracking
8. Returns classification prediction:
   - Outputs `predictions` in standard classification format (compatible with classification blocks)
   - Outputs `error_status` indicating parsing success/failure
   - Outputs `inference_id` with specific type for tracking and lineage

The block enables using VLMs/LLMs for classification by converting their text-based JSON outputs into standardized classification predictions that can be used in workflows like any other classification model output.

## Common Use Cases

- **VLM-Based Classification**: Use Visual Language Models for image classification by parsing VLM outputs into classification predictions (e.g., classify images with VLMs, use GPT-4V for classification, parse Claude Vision classifications), enabling VLM classification workflows
- **LLM Classification Parsing**: Parse LLM text outputs containing classification results into standardized format (e.g., parse GPT classification outputs, convert LLM predictions to classification format, use LLMs for classification), enabling LLM classification workflows
- **Text-to-Classification Conversion**: Convert text-based classification outputs from models into workflow-compatible classification predictions (e.g., convert text predictions to classification format, parse text-based classifications, convert model outputs to classifications), enabling text-to-classification workflows
- **Multi-Format Classification Support**: Handle both single-class and multi-label classification formats from VLM/LLM outputs (e.g., support single-label VLM classifications, support multi-label VLM classifications, handle different classification formats), enabling flexible classification workflows
- **VLM Integration**: Integrate VLM outputs into classification workflows (e.g., use VLMs in classification pipelines, integrate VLM predictions with classification blocks, combine VLM and traditional classification), enabling VLM integration workflows
- **Flexible Classification Sources**: Enable classification from various model types that output text/JSON (e.g., use any text-output model for classification, convert model outputs to classifications, parse various classification formats), enabling flexible classification workflows

## Connecting to Other Blocks

This block receives images and VLM outputs and produces classification predictions:

- **After VLM/LLM blocks** to parse classification outputs into standard format (e.g., VLM output to classification, LLM output to classification, parse model outputs), enabling VLM-to-classification workflows
- **Before classification-based blocks** to use parsed classifications (e.g., use parsed classifications in workflows, provide classifications to downstream blocks, use VLM classifications with classification blocks), enabling classification-to-workflow workflows
- **Before filtering blocks** to filter based on VLM classifications (e.g., filter by VLM classification results, use parsed classifications for filtering, apply filters to VLM predictions), enabling classification-to-filter workflows
- **Before analytics blocks** to analyze VLM classification results (e.g., analyze VLM classifications, perform analytics on parsed classifications, track VLM classification metrics), enabling classification analytics workflows
- **Before visualization blocks** to display VLM classification results (e.g., visualize VLM classifications, display parsed classification predictions, show VLM classification outputs), enabling classification visualization workflows
- **In workflow outputs** to provide VLM classifications as final output (e.g., VLM classification outputs, parsed classification results, VLM-based classification outputs), enabling classification output workflows

## Version Differences

This version (v2) includes the following enhancements over v1:

- **Improved Type System**: The `inference_id` output now uses `INFERENCE_ID_KIND` instead of generic `STRING_KIND`, providing better type safety and semantic clarity for inference ID values in the workflow type system

## Requirements

This block requires an image input (for metadata and dimensions) and a VLM output string containing JSON classification data. The JSON can be raw JSON or wrapped in Markdown code blocks (```json ... ```). The block supports two JSON formats: single-class (with "class_name" and "confidence" fields) and multi-label (with "predicted_classes" array). The `classes` parameter must contain a list of all class names used by the model to generate class_id mappings. Classes are mapped to IDs by index (first class = 0, second = 1, etc.). Classes not in the list get class_id = -1. Confidence scores are normalized to [0.0, 1.0] range. The block outputs classification predictions in standard format (compatible with classification blocks), error_status (boolean), and inference_id (INFERENCE_ID_KIND) for tracking.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/vlm_as_classifier@v2`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `classes` | `List[str]` | List of all class names used by the classification model, in order. Required to generate mapping between class names (from VLM output) and class IDs (for classification format). Classes are mapped to IDs by index: first class = ID 0, second = ID 1, etc. Classes from VLM output that are not in this list get class_id = -1. Should match the classes the VLM was asked to classify.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `VLM As Classifier` in version `v2`.

    - inputs: [`Image Blur`](image_blur.md), [`Image Threshold`](image_threshold.md), [`Morphological Transformation`](morphological_transformation.md), [`Anthropic Claude`](anthropic_claude.md), [`Triangle Visualization`](triangle_visualization.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Qwen-VL`](qwen_vl.md), [`Pixelate Visualization`](pixelate_visualization.md), [`OpenRouter`](open_router.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`Grid Visualization`](grid_visualization.md), [`Mask Visualization`](mask_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`Dynamic Zone`](dynamic_zone.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`Dynamic Crop`](dynamic_crop.md), [`Anthropic Claude`](anthropic_claude.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`QR Code Generator`](qr_code_generator.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Color Visualization`](color_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemini`](google_gemini.md), [`Halo Visualization`](halo_visualization.md), [`Label Visualization`](label_visualization.md), [`SIFT`](sift.md), [`Clip Comparison`](clip_comparison.md), [`Florence-2 Model`](florence2_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Morphological Transformation`](morphological_transformation.md), [`Clip Comparison`](clip_comparison.md), [`Text Display`](text_display.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Google Gemini`](google_gemini.md), [`Camera Calibration`](camera_calibration.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`SIFT Comparison`](sift_comparison.md), [`Google Gemini`](google_gemini.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Corner Visualization`](corner_visualization.md), [`Motion Detection`](motion_detection.md), [`OpenAI`](open_ai.md), [`Depth Estimation`](depth_estimation.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Image Stack`](image_stack.md), [`Image Slicer`](image_slicer.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Google Gemma`](google_gemma.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stitch Images`](stitch_images.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Icon Visualization`](icon_visualization.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Relative Static Crop`](relative_static_crop.md), [`Perspective Correction`](perspective_correction.md), [`Buffer`](buffer.md), [`Crop Visualization`](crop_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Image Slicer`](image_slicer.md), [`Camera Focus`](camera_focus.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`GLM-OCR`](glmocr.md), [`Dimension Collapse`](dimension_collapse.md), [`Size Measurement`](size_measurement.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Image Contours`](image_contours.md), [`Contrast Equalization`](contrast_equalization.md)
    - outputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Triangle Visualization`](triangle_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Slack Notification`](slack_notification.md), [`Email Notification`](email_notification.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Mask Visualization`](mask_visualization.md), [`Template Matching`](template_matching.md), [`Dynamic Zone`](dynamic_zone.md), [`Halo Visualization`](halo_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`SAM 3`](sam3.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Halo Visualization`](halo_visualization.md), [`Label Visualization`](label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Text Display`](text_display.md), [`MQTT Writer`](mqtt_writer.md), [`PLC Writer`](plc_writer.md), [`Gaze Detection`](gaze_detection.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Google Gemini`](google_gemini.md), [`Camera Calibration`](camera_calibration.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`SIFT Comparison`](sift_comparison.md), [`Corner Visualization`](corner_visualization.md), [`Motion Detection`](motion_detection.md), [`Object Detection Model`](object_detection_model.md), [`Time in Zone`](timein_zone.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Time in Zone`](timein_zone.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Image Stack`](image_stack.md), [`Polygon Visualization`](polygon_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Keypoint Visualization`](keypoint_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Perspective Correction`](perspective_correction.md), [`Crop Visualization`](crop_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Email Notification`](email_notification.md), [`Detections Consensus`](detections_consensus.md), [`Trace Visualization`](trace_visualization.md), [`SAM 3`](sam3.md), [`Circle Visualization`](circle_visualization.md), [`Time in Zone`](timein_zone.md), [`Event Writer`](event_writer.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`VLM As Classifier` in version `v2`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Input image that was used to generate the VLM prediction. Used to extract image dimensions (width, height) and metadata (parent_id) for the classification prediction. The same image that was provided to the VLM/LLM block should be used here to maintain consistency..
        - `vlm_output` (*[`language_model_output`](../kinds/language_model_output.md)*): String output from a VLM or LLM block containing classification prediction in JSON format. Can be raw JSON string (e.g., '{"class_name": "dog", "confidence": 0.95}') or JSON wrapped in Markdown code blocks (e.g., ```json {...} ```). Supports two formats: single-class (with 'class_name' and 'confidence' fields) or multi-label (with 'predicted_classes' array). If multiple markdown blocks exist, only the first is parsed..
        - `classes` (*[`list_of_values`](../kinds/list_of_values.md)*): List of all class names used by the classification model, in order. Required to generate mapping between class names (from VLM output) and class IDs (for classification format). Classes are mapped to IDs by index: first class = ID 0, second = ID 1, etc. Classes from VLM output that are not in this list get class_id = -1. Should match the classes the VLM was asked to classify..

    - output
    
        - `error_status` ([`boolean`](../kinds/boolean.md)): Boolean flag.
        - `predictions` ([`classification_prediction`](../kinds/classification_prediction.md)): Predictions from classifier.
        - `inference_id` ([`inference_id`](../kinds/inference_id.md)): Inference identifier.



??? tip "Example JSON definition of step `VLM As Classifier` in version `v2`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/vlm_as_classifier@v2",
	    "image": "$inputs.image",
	    "vlm_output": "$steps.lmm.output",
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
	    ]
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    




## v1

??? "Class: `VLMAsClassifierBlockV1`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/formatters/vlm_as_classifier/v1.py">inference.core.workflows.core_steps.formatters.vlm_as_classifier.v1.VLMAsClassifierBlockV1</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Parse JSON strings from Visual Language Models (VLMs) and Large Language Models (LLMs) into standardized classification prediction format by extracting class predictions, mapping class names to class IDs, handling both single-class and multi-label formats, and converting VLM/LLM text outputs into workflow-compatible classification results for VLM-based classification, LLM classification parsing, and text-to-classification conversion workflows.

## How This Block Works

This block converts VLM/LLM text outputs containing classification predictions into standardized classification prediction format. The block:

1. Receives image and VLM output string containing classification results in JSON format
2. Parses JSON content from VLM output:

   **Handles Markdown-wrapped JSON:**
   - Searches for JSON wrapped in Markdown code blocks (```json ... ```)
   - This format is common in LLM/VLM responses
   - If multiple markdown JSON blocks are found, only the first block is parsed
   - Extracts JSON content from within markdown tags

   **Handles raw JSON strings:**
   - If no markdown blocks are found, attempts to parse the entire string as JSON
   - Supports standard JSON format strings
3. Detects classification format and parses accordingly:

   **Single-Class Classification Format:**
   - Detects format containing "class_name" and "confidence" fields
   - Extracts the predicted class name and confidence score
   - Creates classification prediction with single top class
   - Maps class name to class ID using provided classes list

   **Multi-Label Classification Format:**
   - Detects format containing "predicted_classes" array
   - Extracts all predicted classes with their confidence scores
   - Handles duplicate classes by taking maximum confidence
   - Maps all class names to class IDs using provided classes list
4. Creates class name to class ID mapping:
   - Uses the provided classes list to create index mapping (class_name → class_id)
   - Maps classes in order (first class = ID 0, second = ID 1, etc.)
   - Classes not in the provided list get class_id = -1
5. Normalizes confidence scores:
   - Scales confidence values to valid range [0.0, 1.0]
   - Clamps values outside the range to 0.0 or 1.0
6. Constructs classification prediction:
   - Includes image dimensions (width, height) from input image
   - For single-class: includes "top" class, confidence, and predictions array
   - For multi-label: includes "predicted_classes" list and predictions dictionary
   - Includes inference_id and parent_id for tracking
   - Formats prediction in standard classification prediction format
7. Handles errors:
   - Sets `error_status` to True if JSON parsing fails
   - Sets `error_status` to True if classification format cannot be determined
   - Returns None for predictions when errors occur
   - Always includes inference_id for tracking
8. Returns classification prediction:
   - Outputs `predictions` in standard classification format (compatible with classification blocks)
   - Outputs `error_status` indicating parsing success/failure
   - Outputs `inference_id` for tracking and lineage

The block enables using VLMs/LLMs for classification by converting their text-based JSON outputs into standardized classification predictions that can be used in workflows like any other classification model output.

## Common Use Cases

- **VLM-Based Classification**: Use Visual Language Models for image classification by parsing VLM outputs into classification predictions (e.g., classify images with VLMs, use GPT-4V for classification, parse Claude Vision classifications), enabling VLM classification workflows
- **LLM Classification Parsing**: Parse LLM text outputs containing classification results into standardized format (e.g., parse GPT classification outputs, convert LLM predictions to classification format, use LLMs for classification), enabling LLM classification workflows
- **Text-to-Classification Conversion**: Convert text-based classification outputs from models into workflow-compatible classification predictions (e.g., convert text predictions to classification format, parse text-based classifications, convert model outputs to classifications), enabling text-to-classification workflows
- **Multi-Format Classification Support**: Handle both single-class and multi-label classification formats from VLM/LLM outputs (e.g., support single-label VLM classifications, support multi-label VLM classifications, handle different classification formats), enabling flexible classification workflows
- **VLM Integration**: Integrate VLM outputs into classification workflows (e.g., use VLMs in classification pipelines, integrate VLM predictions with classification blocks, combine VLM and traditional classification), enabling VLM integration workflows
- **Flexible Classification Sources**: Enable classification from various model types that output text/JSON (e.g., use any text-output model for classification, convert model outputs to classifications, parse various classification formats), enabling flexible classification workflows

## Connecting to Other Blocks

This block receives images and VLM outputs and produces classification predictions:

- **After VLM/LLM blocks** to parse classification outputs into standard format (e.g., VLM output to classification, LLM output to classification, parse model outputs), enabling VLM-to-classification workflows
- **Before classification-based blocks** to use parsed classifications (e.g., use parsed classifications in workflows, provide classifications to downstream blocks, use VLM classifications with classification blocks), enabling classification-to-workflow workflows
- **Before filtering blocks** to filter based on VLM classifications (e.g., filter by VLM classification results, use parsed classifications for filtering, apply filters to VLM predictions), enabling classification-to-filter workflows
- **Before analytics blocks** to analyze VLM classification results (e.g., analyze VLM classifications, perform analytics on parsed classifications, track VLM classification metrics), enabling classification analytics workflows
- **Before visualization blocks** to display VLM classification results (e.g., visualize VLM classifications, display parsed classification predictions, show VLM classification outputs), enabling classification visualization workflows
- **In workflow outputs** to provide VLM classifications as final output (e.g., VLM classification outputs, parsed classification results, VLM-based classification outputs), enabling classification output workflows

## Requirements

This block requires an image input (for metadata and dimensions) and a VLM output string containing JSON classification data. The JSON can be raw JSON or wrapped in Markdown code blocks (```json ... ```). The block supports two JSON formats: single-class (with "class_name" and "confidence" fields) and multi-label (with "predicted_classes" array). The `classes` parameter must contain a list of all class names used by the model to generate class_id mappings. Classes are mapped to IDs by index (first class = 0, second = 1, etc.). Classes not in the list get class_id = -1. Confidence scores are normalized to [0.0, 1.0] range. The block outputs classification predictions in standard format (compatible with classification blocks), error_status (boolean), and inference_id (string) for tracking.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/vlm_as_classifier@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `classes` | `List[str]` | List of all class names used by the classification model, in order. Required to generate mapping between class names (from VLM output) and class IDs (for classification format). Classes are mapped to IDs by index: first class = ID 0, second = ID 1, etc. Classes from VLM output that are not in this list get class_id = -1. Should match the classes the VLM was asked to classify.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `VLM As Classifier` in version `v1`.

    - inputs: [`Image Blur`](image_blur.md), [`Image Threshold`](image_threshold.md), [`Morphological Transformation`](morphological_transformation.md), [`Anthropic Claude`](anthropic_claude.md), [`Triangle Visualization`](triangle_visualization.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Qwen-VL`](qwen_vl.md), [`Pixelate Visualization`](pixelate_visualization.md), [`OpenRouter`](open_router.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`Grid Visualization`](grid_visualization.md), [`Mask Visualization`](mask_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`Dynamic Zone`](dynamic_zone.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`Dynamic Crop`](dynamic_crop.md), [`Anthropic Claude`](anthropic_claude.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`QR Code Generator`](qr_code_generator.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Color Visualization`](color_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemini`](google_gemini.md), [`Halo Visualization`](halo_visualization.md), [`Label Visualization`](label_visualization.md), [`SIFT`](sift.md), [`Clip Comparison`](clip_comparison.md), [`Florence-2 Model`](florence2_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Morphological Transformation`](morphological_transformation.md), [`Clip Comparison`](clip_comparison.md), [`Text Display`](text_display.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Google Gemini`](google_gemini.md), [`Camera Calibration`](camera_calibration.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`SIFT Comparison`](sift_comparison.md), [`Google Gemini`](google_gemini.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Corner Visualization`](corner_visualization.md), [`Motion Detection`](motion_detection.md), [`OpenAI`](open_ai.md), [`Depth Estimation`](depth_estimation.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Image Stack`](image_stack.md), [`Image Slicer`](image_slicer.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Google Gemma`](google_gemma.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stitch Images`](stitch_images.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Icon Visualization`](icon_visualization.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Relative Static Crop`](relative_static_crop.md), [`Perspective Correction`](perspective_correction.md), [`Buffer`](buffer.md), [`Crop Visualization`](crop_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Image Slicer`](image_slicer.md), [`Camera Focus`](camera_focus.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`GLM-OCR`](glmocr.md), [`Dimension Collapse`](dimension_collapse.md), [`Size Measurement`](size_measurement.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Image Contours`](image_contours.md), [`Contrast Equalization`](contrast_equalization.md)
    - outputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Threshold`](image_threshold.md), [`Triangle Visualization`](triangle_visualization.md), [`OpenAI`](open_ai.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Qwen-VL`](qwen_vl.md), [`Slack Notification`](slack_notification.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Florence-2 Model`](florence2_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Template Matching`](template_matching.md), [`Dynamic Zone`](dynamic_zone.md), [`Halo Visualization`](halo_visualization.md), [`Webhook Sink`](webhook_sink.md), [`SAM 3`](sam3.md), [`SAM 3`](sam3.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`QR Code Generator`](qr_code_generator.md), [`Google Gemini`](google_gemini.md), [`YOLO-World Model`](yolo_world_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Halo Visualization`](halo_visualization.md), [`Label Visualization`](label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Path Deviation`](path_deviation.md), [`Florence-2 Model`](florence2_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Morphological Transformation`](morphological_transformation.md), [`Text Display`](text_display.md), [`MQTT Writer`](mqtt_writer.md), [`PLC Writer`](plc_writer.md), [`Gaze Detection`](gaze_detection.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Google Gemini`](google_gemini.md), [`Google Gemma API`](google_gemma_api.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Polygon Visualization`](polygon_visualization.md), [`Moondream2`](moondream2.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Google Gemini`](google_gemini.md), [`Motion Detection`](motion_detection.md), [`Time in Zone`](timein_zone.md), [`Time in Zone`](timein_zone.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Polygon Visualization`](polygon_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Preprocessing`](image_preprocessing.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Local File Sink`](local_file_sink.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Email Notification`](email_notification.md), [`Line Counter`](line_counter.md), [`S3 Sink`](s3_sink.md), [`LMM`](lmm.md), [`Distance Measurement`](distance_measurement.md), [`Detections Consensus`](detections_consensus.md), [`Cache Set`](cache_set.md), [`Size Measurement`](size_measurement.md), [`Trace Visualization`](trace_visualization.md), [`SAM 3`](sam3.md), [`Circle Visualization`](circle_visualization.md), [`Time in Zone`](timein_zone.md), [`Event Writer`](event_writer.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Path Deviation`](path_deviation.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Blur`](image_blur.md), [`Google Vision OCR`](google_vision_ocr.md), [`Morphological Transformation`](morphological_transformation.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Email Notification`](email_notification.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`OpenRouter`](open_router.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`OpenAI`](open_ai.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenAI`](open_ai.md), [`LMM For Classification`](lmm_for_classification.md), [`Anthropic Claude`](anthropic_claude.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Object Detection Model`](object_detection_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Cache Get`](cache_get.md), [`Seg Preview`](seg_preview.md), [`Clip Comparison`](clip_comparison.md), [`Camera Calibration`](camera_calibration.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`CogVLM`](cog_vlm.md), [`Corner Visualization`](corner_visualization.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`OpenAI`](open_ai.md), [`Object Detection Model`](object_detection_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Depth Estimation`](depth_estimation.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Image Stack`](image_stack.md), [`Line Counter`](line_counter.md), [`Current Time`](current_time.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Google Gemma`](google_gemma.md), [`Line Counter Visualization`](line_counter_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Keypoint Visualization`](keypoint_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Detections Stitch`](detections_stitch.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Perspective Correction`](perspective_correction.md), [`Crop Visualization`](crop_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`GLM-OCR`](glmocr.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Pixel Color Count`](pixel_color_count.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`VLM As Classifier` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Input image that was used to generate the VLM prediction. Used to extract image dimensions (width, height) and metadata (parent_id) for the classification prediction. The same image that was provided to the VLM/LLM block should be used here to maintain consistency..
        - `vlm_output` (*[`language_model_output`](../kinds/language_model_output.md)*): String output from a VLM or LLM block containing classification prediction in JSON format. Can be raw JSON string (e.g., '{"class_name": "dog", "confidence": 0.95}') or JSON wrapped in Markdown code blocks (e.g., ```json {...} ```). Supports two formats: single-class (with 'class_name' and 'confidence' fields) or multi-label (with 'predicted_classes' array). If multiple markdown blocks exist, only the first is parsed..
        - `classes` (*[`list_of_values`](../kinds/list_of_values.md)*): List of all class names used by the classification model, in order. Required to generate mapping between class names (from VLM output) and class IDs (for classification format). Classes are mapped to IDs by index: first class = ID 0, second = ID 1, etc. Classes from VLM output that are not in this list get class_id = -1. Should match the classes the VLM was asked to classify..

    - output
    
        - `error_status` ([`boolean`](../kinds/boolean.md)): Boolean flag.
        - `predictions` ([`classification_prediction`](../kinds/classification_prediction.md)): Predictions from classifier.
        - `inference_id` ([`string`](../kinds/string.md)): String value.



??? tip "Example JSON definition of step `VLM As Classifier` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/vlm_as_classifier@v1",
	    "image": "$inputs.image",
	    "vlm_output": "$steps.lmm.output",
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
	    ]
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    


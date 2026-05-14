
# Expression



??? "Class: `ExpressionBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/formatters/expression/v1.py">inference.core.workflows.core_steps.formatters.expression.v1.ExpressionBlockV1</a>
    



Create conditional logic and business rules in workflows using switch-case statements that evaluate conditions on input variables, optionally transform data with operations, and return different outputs based on which condition matches, enabling conditional execution, business logic implementation, rule-based decision making, and dynamic output generation workflows.

## How This Block Works

This block implements conditional logic similar to switch-case or if-else-if statements in programming. The block:

1. Receives input data as a dictionary of named variables from workflow steps
2. Optionally applies data transformations using operations:
   - Performs operations on data variables before condition evaluation
   - Uses the same operation system as Property Definition block
   - Transforms data (e.g., extract properties, filter, select) to prepare variables for conditions
   - Stores transformed values as variables for use in conditions
3. Evaluates switch-case statements sequentially:
   - Tests each case condition in order until one matches
   - Stops at the first matching case and returns its result
   - If no case matches, returns the default result
4. Evaluates conditions using a flexible expression system:
   - **Binary Statements**: Compare two values using operators (==, !=, >, <, >=, <=, contains, startsWith, endsWith, in, any in, all in)
   - **Unary Statements**: Test single values (Exists, DoesNotExist, is True, is False, is empty, is not empty)
   - **Statement Groups**: Combine multiple statements with AND/OR operators for complex conditions
   - Conditions can reference variables by name (DynamicOperand) or use literal values (StaticOperand)
5. Returns results based on matched case:

   **Static Results:**
   - Returns a fixed value defined in the case (e.g., "PASS", "FAIL", numeric values, strings)

   **Dynamic Results:**
   - Returns a value from a variable (can reference any input variable)
   - Optionally applies operations to transform the variable before returning
   - Enables returning computed or extracted values as output

6. Handles default case:
   - If no case condition matches, returns the default result
   - Default can be static or dynamic, just like case results

The block enables complex conditional logic by combining data transformation operations with flexible condition evaluation. Conditions can compare variables, test existence, check membership, perform string operations, and combine multiple conditions with logical operators. This makes it powerful for implementing business rules, validation logic, classification based on multiple criteria, and conditional data transformation.

## Common Use Cases

- **Business Logic Implementation**: Implement conditional business rules and validation logic (e.g., validate detection matches reference, implement quality checks, enforce business rules), enabling business logic workflows
- **Conditional Classification**: Classify data based on multiple conditions and criteria (e.g., classify detections based on properties, categorize results by conditions, implement multi-criteria classification), enabling conditional classification workflows
- **Validation and Quality Control**: Validate data or predictions against reference values or thresholds (e.g., validate predictions match expected classes, check quality thresholds, verify compliance), enabling validation workflows
- **Rule-Based Decision Making**: Make decisions based on complex rule sets (e.g., approve/reject based on multiple criteria, route data based on conditions, make decisions using rule sets), enabling rule-based decision workflows
- **Dynamic Output Generation**: Generate different outputs based on input conditions (e.g., return different values based on conditions, generate conditional outputs, create dynamic results), enabling dynamic output workflows
- **Multi-Condition Filtering**: Implement complex filtering logic with multiple conditions (e.g., filter based on multiple criteria, apply complex conditional filters, implement multi-factor filtering), enabling conditional filtering workflows

## Connecting to Other Blocks

This block receives data from workflow steps and produces conditional output:

- **After model or analytics blocks** to implement conditional logic on predictions or results (e.g., validate predictions, classify results, apply conditional rules), enabling conditional logic workflows
- **After Property Definition blocks** to use extracted properties in conditions (e.g., use extracted values in conditions, compare extracted properties, implement logic on extracted data), enabling property-to-condition workflows
- **Before logic blocks** like Continue If to provide conditional inputs (e.g., provide conditional values for filtering, supply conditional inputs for decisions), enabling expression-to-logic workflows
- **Before data storage blocks** to conditionally format or transform data for storage (e.g., conditionally format for storage, apply conditional transformations, prepare conditional outputs), enabling conditional storage workflows
- **Before notification blocks** to send conditional notifications (e.g., send conditional alerts, notify based on conditions, trigger conditional notifications), enabling conditional notification workflows
- **In workflow outputs** to provide conditional final outputs (e.g., conditional workflow outputs, dynamic result generation, conditional output formatting), enabling conditional output workflows

## Requirements

This block requires input data as a dictionary where keys are variable names and values are data from workflow steps. The switch parameter defines cases with conditions and results. Conditions support binary comparisons (==, !=, >, <, >=, <=, contains, in, etc.), unary tests (Exists, is empty, etc.), and logical combinations (AND/OR). Data operations are optional and use the same operation system as Property Definition block. The block evaluates cases in order and returns the result of the first matching case, or the default result if no cases match. Results can be static values or dynamic values from variables (optionally with operations applied).


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/expression@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `data_operations` | `Dict[str, List[Union[ClassificationPropertyExtract, ConvertDictionaryToJSON, ConvertImageToBase64, ConvertImageToJPEG, DetectionsFilter, DetectionsOffset, DetectionsPropertyExtract, DetectionsRename, DetectionsSelection, DetectionsShift, DetectionsToDictionary, Divide, ExtractDetectionProperty, ExtractFrameMetadata, ExtractImageProperty, LookupTable, Multiply, NumberRound, NumericSequenceAggregate, PickDetectionsByParentClass, RandomNumber, SequenceAggregate, SequenceApply, SequenceElementsCount, SequenceLength, SequenceMap, SortDetections, StringMatches, StringSubSequence, StringToLowerCase, StringToUpperCase, TimestampToISOFormat, ToBoolean, ToNumber, ToString]]]` | Optional dictionary of operations to transform data variables before condition evaluation. Keys are variable names from data, values are lists of operations (same as Property Definition block). Operations are applied to transform variables before they are used in conditions. Useful for extracting properties, filtering, or transforming data before evaluation. Empty dictionary (default) means no transformations are applied.. | ❌ |
| `switch` | `CasesDefinition` | Switch-case logic definition containing cases with conditions and results. Each case has a condition (StatementGroup with binary/unary statements) and a result (static value or dynamic variable). Cases are evaluated in order - first matching case's result is returned. Default result is returned if no cases match. Supports complex conditions with AND/OR operators, comparison operators (==, !=, >, <, >=, <=), string operations (contains, startsWith, endsWith), membership tests (in, any in, all in), and existence tests (Exists, is empty).. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Expression` in version `v1`.

    - inputs: [`Triangle Visualization`](triangle_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Florence-2 Model`](florence2_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Image Stack`](image_stack.md), [`Path Deviation`](path_deviation.md), [`Distance Measurement`](distance_measurement.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`Dominant Color`](dominant_color.md), [`Background Subtraction`](background_subtraction.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Camera Focus`](camera_focus.md), [`SmolVLM2`](smol_vlm2.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Google Gemini`](google_gemini.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Cache Get`](cache_get.md), [`Email Notification`](email_notification.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Inner Workflow`](inner_workflow.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Email Notification`](email_notification.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Contours`](image_contours.md), [`Icon Visualization`](icon_visualization.md), [`Overlap Filter`](overlap_filter.md), [`Circle Visualization`](circle_visualization.md), [`Buffer`](buffer.md), [`Mask Visualization`](mask_visualization.md), [`OpenAI`](open_ai.md), [`Reference Path Visualization`](reference_path_visualization.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Velocity`](velocity.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Time in Zone`](timein_zone.md), [`Image Slicer`](image_slicer.md), [`Qwen3.5`](qwen3.5.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Polygon Visualization`](polygon_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`JSON Parser`](json_parser.md), [`SAM 3`](sam3.md), [`QR Code Detection`](qr_code_detection.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Detections Stitch`](detections_stitch.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Grid Visualization`](grid_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Color Visualization`](color_visualization.md), [`Time in Zone`](timein_zone.md), [`VLM As Detector`](vlm_as_detector.md), [`Dot Visualization`](dot_visualization.md), [`Size Measurement`](size_measurement.md), [`OCR Model`](ocr_model.md), [`Qwen3-VL`](qwen3_vl.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Trace Visualization`](trace_visualization.md), [`Slack Notification`](slack_notification.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Moondream2`](moondream2.md), [`Byte Tracker`](byte_tracker.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Template Matching`](template_matching.md), [`Google Gemma API`](google_gemma_api.md), [`Anthropic Claude`](anthropic_claude.md), [`LMM For Classification`](lmm_for_classification.md), [`VLM As Classifier`](vlm_as_classifier.md), [`OpenAI`](open_ai.md), [`Byte Tracker`](byte_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Webhook Sink`](webhook_sink.md), [`Mask Edge Snap`](mask_edge_snap.md), [`GLM-OCR`](glmocr.md), [`Background Color Visualization`](background_color_visualization.md), [`Time in Zone`](timein_zone.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Detections Transformation`](detections_transformation.md), [`SORT Tracker`](sort_tracker.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Stitch Images`](stitch_images.md), [`S3 Sink`](s3_sink.md), [`Google Vision OCR`](google_vision_ocr.md), [`Detections Combine`](detections_combine.md), [`Image Blur`](image_blur.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Byte Tracker`](byte_tracker.md), [`Morphological Transformation`](morphological_transformation.md), [`Rate Limiter`](rate_limiter.md), [`Cosine Similarity`](cosine_similarity.md), [`Identify Changes`](identify_changes.md), [`QR Code Generator`](qr_code_generator.md), [`Image Preprocessing`](image_preprocessing.md), [`Detection Event Log`](detection_event_log.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Barcode Detection`](barcode_detection.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Crop Visualization`](crop_visualization.md), [`Line Counter`](line_counter.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Delta Filter`](delta_filter.md), [`Relative Static Crop`](relative_static_crop.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Identify Outliers`](identify_outliers.md), [`Perspective Correction`](perspective_correction.md), [`Halo Visualization`](halo_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Seg Preview`](seg_preview.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Continue If`](continue_if.md), [`Line Counter`](line_counter.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Text Display`](text_display.md), [`SIFT`](sift.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Gaze Detection`](gaze_detection.md), [`Data Aggregator`](data_aggregator.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Camera Focus`](camera_focus.md), [`Anthropic Claude`](anthropic_claude.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Motion Detection`](motion_detection.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Property Definition`](property_definition.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Image Threshold`](image_threshold.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`Blur Visualization`](blur_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Corner Visualization`](corner_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Label Visualization`](label_visualization.md), [`Detections Consensus`](detections_consensus.md), [`CSV Formatter`](csv_formatter.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detections Merge`](detections_merge.md), [`Detection Offset`](detection_offset.md), [`Camera Calibration`](camera_calibration.md), [`SAM 3`](sam3.md), [`Environment Secrets Store`](environment_secrets_store.md), [`EasyOCR`](easy_ocr.md), [`Morphological Transformation`](morphological_transformation.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Path Deviation`](path_deviation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`CogVLM`](cog_vlm.md), [`LMM`](lmm.md), [`Depth Estimation`](depth_estimation.md), [`SIFT Comparison`](sift_comparison.md), [`Detections Filter`](detections_filter.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Expression`](expression.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Pixel Color Count`](pixel_color_count.md), [`Local File Sink`](local_file_sink.md), [`Cache Set`](cache_set.md), [`Bounding Rectangle`](bounding_rectangle.md)
    - outputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`Triangle Visualization`](triangle_visualization.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Florence-2 Model`](florence2_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Path Deviation`](path_deviation.md), [`Distance Measurement`](distance_measurement.md), [`Image Stack`](image_stack.md), [`OpenAI`](open_ai.md), [`Dominant Color`](dominant_color.md), [`Clip Comparison`](clip_comparison.md), [`Background Subtraction`](background_subtraction.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Camera Focus`](camera_focus.md), [`SmolVLM2`](smol_vlm2.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Google Gemini`](google_gemini.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Cache Get`](cache_get.md), [`Email Notification`](email_notification.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Inner Workflow`](inner_workflow.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Email Notification`](email_notification.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Contours`](image_contours.md), [`Icon Visualization`](icon_visualization.md), [`Overlap Filter`](overlap_filter.md), [`Circle Visualization`](circle_visualization.md), [`Mask Visualization`](mask_visualization.md), [`OpenAI`](open_ai.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Velocity`](velocity.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Buffer`](buffer.md), [`Time in Zone`](timein_zone.md), [`Image Slicer`](image_slicer.md), [`Qwen3.5`](qwen3.5.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Polygon Visualization`](polygon_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`SAM 3`](sam3.md), [`JSON Parser`](json_parser.md), [`QR Code Detection`](qr_code_detection.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Detections Stitch`](detections_stitch.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Grid Visualization`](grid_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Color Visualization`](color_visualization.md), [`Cache Set`](cache_set.md), [`Time in Zone`](timein_zone.md), [`VLM As Detector`](vlm_as_detector.md), [`Dot Visualization`](dot_visualization.md), [`Size Measurement`](size_measurement.md), [`OCR Model`](ocr_model.md), [`Qwen3-VL`](qwen3_vl.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Trace Visualization`](trace_visualization.md), [`Slack Notification`](slack_notification.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Moondream2`](moondream2.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Byte Tracker`](byte_tracker.md), [`Template Matching`](template_matching.md), [`Google Gemma API`](google_gemma_api.md), [`LMM For Classification`](lmm_for_classification.md), [`Anthropic Claude`](anthropic_claude.md), [`VLM As Classifier`](vlm_as_classifier.md), [`OpenAI`](open_ai.md), [`Byte Tracker`](byte_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Webhook Sink`](webhook_sink.md), [`GLM-OCR`](glmocr.md), [`Background Color Visualization`](background_color_visualization.md), [`Time in Zone`](timein_zone.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Detections Transformation`](detections_transformation.md), [`SORT Tracker`](sort_tracker.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Stitch Images`](stitch_images.md), [`S3 Sink`](s3_sink.md), [`Detections Combine`](detections_combine.md), [`Google Vision OCR`](google_vision_ocr.md), [`Image Blur`](image_blur.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Byte Tracker`](byte_tracker.md), [`Morphological Transformation`](morphological_transformation.md), [`Identify Changes`](identify_changes.md), [`Cosine Similarity`](cosine_similarity.md), [`QR Code Generator`](qr_code_generator.md), [`Rate Limiter`](rate_limiter.md), [`Image Preprocessing`](image_preprocessing.md), [`Detection Event Log`](detection_event_log.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Barcode Detection`](barcode_detection.md), [`Line Counter`](line_counter.md), [`Crop Visualization`](crop_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Delta Filter`](delta_filter.md), [`Relative Static Crop`](relative_static_crop.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Identify Outliers`](identify_outliers.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Perspective Correction`](perspective_correction.md), [`Halo Visualization`](halo_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Object Detection Model`](object_detection_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Seg Preview`](seg_preview.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Continue If`](continue_if.md), [`Line Counter`](line_counter.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Text Display`](text_display.md), [`SIFT`](sift.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Gaze Detection`](gaze_detection.md), [`Data Aggregator`](data_aggregator.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Camera Focus`](camera_focus.md), [`Anthropic Claude`](anthropic_claude.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Motion Detection`](motion_detection.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Property Definition`](property_definition.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Image Threshold`](image_threshold.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Dimension Collapse`](dimension_collapse.md), [`Blur Visualization`](blur_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Label Visualization`](label_visualization.md), [`Detections Consensus`](detections_consensus.md), [`CSV Formatter`](csv_formatter.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detections Merge`](detections_merge.md), [`Detection Offset`](detection_offset.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Calibration`](camera_calibration.md), [`SAM 3`](sam3.md), [`EasyOCR`](easy_ocr.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Path Deviation`](path_deviation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`SIFT Comparison`](sift_comparison.md), [`LMM`](lmm.md), [`Depth Estimation`](depth_estimation.md), [`Anthropic Claude`](anthropic_claude.md), [`Detections Filter`](detections_filter.md), [`CogVLM`](cog_vlm.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Expression`](expression.md), [`Pixel Color Count`](pixel_color_count.md), [`Local File Sink`](local_file_sink.md), [`Line Counter Visualization`](line_counter_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Expression` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `data` (*[`*`](../kinds/wildcard.md)*): Dictionary of named variables containing data from workflow steps. Variable names are used in conditions and results. Keys are variable names, values are selectors referencing workflow step outputs. Variables can be referenced in conditions and dynamic results. Example: {'predictions': '$steps.model.predictions', 'reference': '$inputs.reference_class_names'} creates variables 'predictions' and 'reference'..

    - output
    
        - `output` ([`*`](../kinds/wildcard.md)): Equivalent of any element.



??? tip "Example JSON definition of step `Expression` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/expression@v1",
	    "data": {
	        "predictions": "$steps.model.predictions",
	        "reference": "$inputs.reference_class_names"
	    },
	    "data_operations": {
	        "predictions": [
	            {
	                "property_name": "class_name",
	                "type": "DetectionsPropertyExtract"
	            }
	        ]
	    },
	    "switch": {
	        "cases": [
	            {
	                "condition": {
	                    "statements": [
	                        {
	                            "comparator": {
	                                "type": "=="
	                            },
	                            "left_operand": {
	                                "operand_name": "class_name",
	                                "type": "DynamicOperand"
	                            },
	                            "right_operand": {
	                                "operand_name": "reference",
	                                "type": "DynamicOperand"
	                            },
	                            "type": "BinaryStatement"
	                        }
	                    ],
	                    "type": "StatementGroup"
	                },
	                "result": {
	                    "type": "StaticCaseResult",
	                    "value": "PASS"
	                },
	                "type": "CaseDefinition"
	            }
	        ],
	        "default": {
	            "type": "StaticCaseResult",
	            "value": "FAIL"
	        },
	        "type": "CasesDefinition"
	    }
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    


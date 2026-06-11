
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

    - inputs: [`Google Gemma API`](google_gemma_api.md), [`Camera Focus`](camera_focus.md), [`Event Writer`](event_writer.md), [`Google Gemini`](google_gemini.md), [`Rate Limiter`](rate_limiter.md), [`Current Time`](current_time.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Camera Focus`](camera_focus.md), [`Inner Workflow`](inner_workflow.md), [`Image Contours`](image_contours.md), [`Image Preprocessing`](image_preprocessing.md), [`Background Subtraction`](background_subtraction.md), [`SAM 3`](sam3.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Color Visualization`](color_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`YOLO-World Model`](yolo_world_model.md), [`Seg Preview`](seg_preview.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`JSON Parser`](json_parser.md), [`Cache Set`](cache_set.md), [`Relative Static Crop`](relative_static_crop.md), [`Google Vision OCR`](google_vision_ocr.md), [`SIFT Comparison`](sift_comparison.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`EasyOCR`](easy_ocr.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`VLM As Detector`](vlm_as_detector.md), [`SAM 3`](sam3.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Qwen3.5`](qwen3.5.md), [`Google Gemma`](google_gemma.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Blur`](image_blur.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Identify Changes`](identify_changes.md), [`Buffer`](buffer.md), [`MQTT Writer`](mqtt_writer.md), [`Anthropic Claude`](anthropic_claude.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Camera Calibration`](camera_calibration.md), [`VLM As Detector`](vlm_as_detector.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`SIFT Comparison`](sift_comparison.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Expression`](expression.md), [`Detections Stitch`](detections_stitch.md), [`Overlap Analysis`](overlap_analysis.md), [`LMM For Classification`](lmm_for_classification.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Barcode Detection`](barcode_detection.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Perspective Correction`](perspective_correction.md), [`Cache Get`](cache_get.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`SAM 3`](sam3.md), [`Time in Zone`](timein_zone.md), [`Detection Offset`](detection_offset.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Contrast Equalization`](contrast_equalization.md), [`Clip Comparison`](clip_comparison.md), [`SORT Tracker`](sort_tracker.md), [`Qwen-VL`](qwen_vl.md), [`Image Stack`](image_stack.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Identify Outliers`](identify_outliers.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Dot Visualization`](dot_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`OpenAI`](open_ai.md), [`Continue If`](continue_if.md), [`Moondream2`](moondream2.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Size Measurement`](size_measurement.md), [`QR Code Detection`](qr_code_detection.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Dominant Color`](dominant_color.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Property Definition`](property_definition.md), [`Path Deviation`](path_deviation.md), [`Webhook Sink`](webhook_sink.md), [`SIFT`](sift.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Image Slicer`](image_slicer.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Halo Visualization`](halo_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`Email Notification`](email_notification.md), [`QR Code Generator`](qr_code_generator.md), [`Local File Sink`](local_file_sink.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Object Detection Model`](object_detection_model.md), [`SmolVLM2`](smol_vlm2.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Path Deviation`](path_deviation.md), [`Template Matching`](template_matching.md), [`Image Slicer`](image_slicer.md), [`Depth Estimation`](depth_estimation.md), [`Text Display`](text_display.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Byte Tracker`](byte_tracker.md), [`Distance Measurement`](distance_measurement.md), [`Object Detection Model`](object_detection_model.md), [`Time in Zone`](timein_zone.md), [`Icon Visualization`](icon_visualization.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Anthropic Claude`](anthropic_claude.md), [`Cosine Similarity`](cosine_similarity.md), [`Grid Visualization`](grid_visualization.md), [`Qwen3-VL`](qwen3_vl.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Anthropic Claude`](anthropic_claude.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenRouter`](open_router.md), [`LMM`](lmm.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Image Threshold`](image_threshold.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Overlap Filter`](overlap_filter.md), [`Data Aggregator`](data_aggregator.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Label Visualization`](label_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Morphological Transformation`](morphological_transformation.md), [`Morphological Transformation`](morphological_transformation.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Dynamic Crop`](dynamic_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`Detections Transformation`](detections_transformation.md), [`CSV Formatter`](csv_formatter.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Gaze Detection`](gaze_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Byte Tracker`](byte_tracker.md), [`Absolute Static Crop`](absolute_static_crop.md), [`CogVLM`](cog_vlm.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Stitch Images`](stitch_images.md), [`Florence-2 Model`](florence2_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Line Counter`](line_counter.md), [`Detections Merge`](detections_merge.md), [`Clip Comparison`](clip_comparison.md), [`Time in Zone`](timein_zone.md), [`Byte Tracker`](byte_tracker.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Background Color Visualization`](background_color_visualization.md), [`Detections Consensus`](detections_consensus.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Corner Visualization`](corner_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Delta Filter`](delta_filter.md), [`Detections Combine`](detections_combine.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Halo Visualization`](halo_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Line Counter`](line_counter.md), [`OCR Model`](ocr_model.md), [`OpenAI`](open_ai.md), [`Email Notification`](email_notification.md), [`Slack Notification`](slack_notification.md), [`S3 Sink`](s3_sink.md), [`GLM-OCR`](glmocr.md), [`Detection Event Log`](detection_event_log.md), [`OpenAI`](open_ai.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Detections Filter`](detections_filter.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Velocity`](velocity.md)
    - outputs: [`Google Gemma API`](google_gemma_api.md), [`Camera Focus`](camera_focus.md), [`Event Writer`](event_writer.md), [`Google Gemini`](google_gemini.md), [`Rate Limiter`](rate_limiter.md), [`Current Time`](current_time.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Camera Focus`](camera_focus.md), [`Inner Workflow`](inner_workflow.md), [`Image Preprocessing`](image_preprocessing.md), [`Background Subtraction`](background_subtraction.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SAM 3`](sam3.md), [`Image Contours`](image_contours.md), [`Color Visualization`](color_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`YOLO-World Model`](yolo_world_model.md), [`Seg Preview`](seg_preview.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`JSON Parser`](json_parser.md), [`Cache Set`](cache_set.md), [`Google Vision OCR`](google_vision_ocr.md), [`Relative Static Crop`](relative_static_crop.md), [`SIFT Comparison`](sift_comparison.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Object Detection Model`](object_detection_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`EasyOCR`](easy_ocr.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`VLM As Detector`](vlm_as_detector.md), [`SAM 3`](sam3.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Qwen3.5`](qwen3.5.md), [`Google Gemma`](google_gemma.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Identify Changes`](identify_changes.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Image Blur`](image_blur.md), [`MQTT Writer`](mqtt_writer.md), [`Buffer`](buffer.md), [`Anthropic Claude`](anthropic_claude.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Camera Calibration`](camera_calibration.md), [`VLM As Detector`](vlm_as_detector.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`SIFT Comparison`](sift_comparison.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Expression`](expression.md), [`Detections Stitch`](detections_stitch.md), [`LMM For Classification`](lmm_for_classification.md), [`Overlap Analysis`](overlap_analysis.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Barcode Detection`](barcode_detection.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Perspective Correction`](perspective_correction.md), [`Cache Get`](cache_get.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`SAM 3`](sam3.md), [`Time in Zone`](timein_zone.md), [`Detection Offset`](detection_offset.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Mask Edge Snap`](mask_edge_snap.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Contrast Equalization`](contrast_equalization.md), [`Clip Comparison`](clip_comparison.md), [`SORT Tracker`](sort_tracker.md), [`Qwen-VL`](qwen_vl.md), [`Image Stack`](image_stack.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Identify Outliers`](identify_outliers.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Dot Visualization`](dot_visualization.md), [`OpenAI`](open_ai.md), [`Dynamic Zone`](dynamic_zone.md), [`Continue If`](continue_if.md), [`Moondream2`](moondream2.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Size Measurement`](size_measurement.md), [`QR Code Detection`](qr_code_detection.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Dominant Color`](dominant_color.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Path Deviation`](path_deviation.md), [`Property Definition`](property_definition.md), [`Webhook Sink`](webhook_sink.md), [`SIFT`](sift.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Image Slicer`](image_slicer.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Halo Visualization`](halo_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`Email Notification`](email_notification.md), [`QR Code Generator`](qr_code_generator.md), [`Local File Sink`](local_file_sink.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Object Detection Model`](object_detection_model.md), [`SmolVLM2`](smol_vlm2.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Path Deviation`](path_deviation.md), [`Template Matching`](template_matching.md), [`Image Slicer`](image_slicer.md), [`Depth Estimation`](depth_estimation.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Text Display`](text_display.md), [`Byte Tracker`](byte_tracker.md), [`Time in Zone`](timein_zone.md), [`Object Detection Model`](object_detection_model.md), [`Distance Measurement`](distance_measurement.md), [`Icon Visualization`](icon_visualization.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Anthropic Claude`](anthropic_claude.md), [`Grid Visualization`](grid_visualization.md), [`Qwen3-VL`](qwen3_vl.md), [`OpenRouter`](open_router.md), [`Cosine Similarity`](cosine_similarity.md), [`Anthropic Claude`](anthropic_claude.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`LMM`](lmm.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Image Threshold`](image_threshold.md), [`Overlap Filter`](overlap_filter.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Data Aggregator`](data_aggregator.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Label Visualization`](label_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Morphological Transformation`](morphological_transformation.md), [`Morphological Transformation`](morphological_transformation.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Dynamic Crop`](dynamic_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`Detections Transformation`](detections_transformation.md), [`CSV Formatter`](csv_formatter.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Gaze Detection`](gaze_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Byte Tracker`](byte_tracker.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Absolute Static Crop`](absolute_static_crop.md), [`CogVLM`](cog_vlm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Stitch Images`](stitch_images.md), [`Florence-2 Model`](florence2_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Detections Merge`](detections_merge.md), [`Clip Comparison`](clip_comparison.md), [`Byte Tracker`](byte_tracker.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Detections Consensus`](detections_consensus.md), [`Background Color Visualization`](background_color_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Corner Visualization`](corner_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Delta Filter`](delta_filter.md), [`Detections Combine`](detections_combine.md), [`Line Counter Visualization`](line_counter_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Halo Visualization`](halo_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Line Counter`](line_counter.md), [`OCR Model`](ocr_model.md), [`OpenAI`](open_ai.md), [`Email Notification`](email_notification.md), [`Slack Notification`](slack_notification.md), [`S3 Sink`](s3_sink.md), [`GLM-OCR`](glmocr.md), [`Detection Event Log`](detection_event_log.md), [`OpenAI`](open_ai.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Detections Filter`](detections_filter.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Velocity`](velocity.md)

    
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



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

    - inputs: [`Cache Set`](cache_set.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Path Deviation`](path_deviation.md), [`Image Blur`](image_blur.md), [`SmolVLM2`](smol_vlm2.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Overlap Filter`](overlap_filter.md), [`SIFT Comparison`](sift_comparison.md), [`Reference Path Visualization`](reference_path_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Event Writer`](event_writer.md), [`Slack Notification`](slack_notification.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Halo Visualization`](halo_visualization.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Image Stack`](image_stack.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemma`](google_gemma.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Object Detection Model`](object_detection_model.md), [`Dot Visualization`](dot_visualization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Label Visualization`](label_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Rate Limiter`](rate_limiter.md), [`Velocity`](velocity.md), [`Pixelate Visualization`](pixelate_visualization.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Qwen3-VL`](qwen3_vl.md), [`Google Gemini`](google_gemini.md), [`JSON Parser`](json_parser.md), [`Anthropic Claude`](anthropic_claude.md), [`Track Class Lock`](track_class_lock.md), [`Cache Get`](cache_get.md), [`OpenAI`](open_ai.md), [`Trace Visualization`](trace_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detection Event Log`](detection_event_log.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Clip Comparison`](clip_comparison.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`Qwen3.5`](qwen3.5.md), [`GLM-OCR`](glmocr.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Buffer`](buffer.md), [`QR Code Detection`](qr_code_detection.md), [`SIFT Comparison`](sift_comparison.md), [`CSV Formatter`](csv_formatter.md), [`Webhook Sink`](webhook_sink.md), [`MQTT Writer`](mqtt_writer.md), [`Image Contours`](image_contours.md), [`Motion Detection`](motion_detection.md), [`Local File Sink`](local_file_sink.md), [`Google Gemini`](google_gemini.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Polygon Visualization`](polygon_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`SIFT`](sift.md), [`Classification Label Visualization`](classification_label_visualization.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Template Matching`](template_matching.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Icon Visualization`](icon_visualization.md), [`Seg Preview`](seg_preview.md), [`Dynamic Crop`](dynamic_crop.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Detections Transformation`](detections_transformation.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Crop Visualization`](crop_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Continue If`](continue_if.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Byte Tracker`](byte_tracker.md), [`Mask Visualization`](mask_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Delta Filter`](delta_filter.md), [`Expression`](expression.md), [`Detection Offset`](detection_offset.md), [`Detections Stitch`](detections_stitch.md), [`Distance Measurement`](distance_measurement.md), [`Barcode Detection`](barcode_detection.md), [`SORT Tracker`](sort_tracker.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Anthropic Claude`](anthropic_claude.md), [`Text Display`](text_display.md), [`Morphological Transformation`](morphological_transformation.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Inner Workflow`](inner_workflow.md), [`Overlap Analysis`](overlap_analysis.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`VLM As Detector`](vlm_as_detector.md), [`Detections Consensus`](detections_consensus.md), [`Object Detection Model`](object_detection_model.md), [`Detections Filter`](detections_filter.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Detections Merge`](detections_merge.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Time in Zone`](timein_zone.md), [`SAM 3`](sam3.md), [`Size Measurement`](size_measurement.md), [`Circle Visualization`](circle_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Path Deviation`](path_deviation.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Email Notification`](email_notification.md), [`S3 Sink`](s3_sink.md), [`Camera Focus`](camera_focus.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Identify Changes`](identify_changes.md), [`Byte Tracker`](byte_tracker.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SAM 3`](sam3.md), [`Image Slicer`](image_slicer.md), [`LMM For Classification`](lmm_for_classification.md), [`Dominant Color`](dominant_color.md), [`OCR Model`](ocr_model.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Switch Case`](switch_case.md), [`OpenAI`](open_ai.md), [`Google Gemma API`](google_gemma_api.md), [`Stitch Images`](stitch_images.md), [`Identify Outliers`](identify_outliers.md), [`Time in Zone`](timein_zone.md), [`Morphological Transformation`](morphological_transformation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`EasyOCR`](easy_ocr.md), [`YOLO-World Model`](yolo_world_model.md), [`Current Time`](current_time.md), [`Blur Visualization`](blur_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Moondream2`](moondream2.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`Corner Visualization`](corner_visualization.md), [`OpenRouter`](open_router.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Pixel Color Count`](pixel_color_count.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`SAM 3`](sam3.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Google Vision OCR`](google_vision_ocr.md), [`Image Threshold`](image_threshold.md), [`Byte Tracker`](byte_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`LMM`](lmm.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Time in Zone`](timein_zone.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Line Counter`](line_counter.md), [`Line Counter Visualization`](line_counter_visualization.md), [`CogVLM`](cog_vlm.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Relative Static Crop`](relative_static_crop.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Grid Visualization`](grid_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Property Definition`](property_definition.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Gaze Detection`](gaze_detection.md), [`Anthropic Claude`](anthropic_claude.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Color Visualization`](color_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Detections Combine`](detections_combine.md), [`Triangle Visualization`](triangle_visualization.md), [`Data Aggregator`](data_aggregator.md), [`QR Code Generator`](qr_code_generator.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Background Subtraction`](background_subtraction.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Qwen-VL`](qwen_vl.md), [`Florence-2 Model`](florence2_model.md), [`Perspective Correction`](perspective_correction.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Cosine Similarity`](cosine_similarity.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Depth Estimation`](depth_estimation.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Contrast Equalization`](contrast_equalization.md), [`Camera Calibration`](camera_calibration.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`VLM As Detector`](vlm_as_detector.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Line Counter`](line_counter.md), [`Object Detection Model`](object_detection_model.md)
    - outputs: [`Cache Set`](cache_set.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Path Deviation`](path_deviation.md), [`Image Blur`](image_blur.md), [`SmolVLM2`](smol_vlm2.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Overlap Filter`](overlap_filter.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`SIFT Comparison`](sift_comparison.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Event Writer`](event_writer.md), [`Slack Notification`](slack_notification.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Halo Visualization`](halo_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Stack`](image_stack.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Clip Comparison`](clip_comparison.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Google Gemma`](google_gemma.md), [`Object Detection Model`](object_detection_model.md), [`Dot Visualization`](dot_visualization.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Label Visualization`](label_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Rate Limiter`](rate_limiter.md), [`Velocity`](velocity.md), [`Pixelate Visualization`](pixelate_visualization.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Qwen3-VL`](qwen3_vl.md), [`Google Gemini`](google_gemini.md), [`JSON Parser`](json_parser.md), [`Anthropic Claude`](anthropic_claude.md), [`Track Class Lock`](track_class_lock.md), [`Cache Get`](cache_get.md), [`OpenAI`](open_ai.md), [`Trace Visualization`](trace_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detection Event Log`](detection_event_log.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`OpenAI`](open_ai.md), [`Camera Focus`](camera_focus.md), [`Clip Comparison`](clip_comparison.md), [`Qwen3.5`](qwen3.5.md), [`GLM-OCR`](glmocr.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Buffer`](buffer.md), [`MQTT Writer`](mqtt_writer.md), [`Webhook Sink`](webhook_sink.md), [`SIFT Comparison`](sift_comparison.md), [`CSV Formatter`](csv_formatter.md), [`QR Code Detection`](qr_code_detection.md), [`Image Contours`](image_contours.md), [`Motion Detection`](motion_detection.md), [`Local File Sink`](local_file_sink.md), [`Google Gemini`](google_gemini.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Polygon Visualization`](polygon_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`SIFT`](sift.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Template Matching`](template_matching.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Icon Visualization`](icon_visualization.md), [`Seg Preview`](seg_preview.md), [`Dynamic Crop`](dynamic_crop.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Crop Visualization`](crop_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Continue If`](continue_if.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Byte Tracker`](byte_tracker.md), [`Mask Visualization`](mask_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Delta Filter`](delta_filter.md), [`Detections Stitch`](detections_stitch.md), [`Distance Measurement`](distance_measurement.md), [`Detection Offset`](detection_offset.md), [`SORT Tracker`](sort_tracker.md), [`Barcode Detection`](barcode_detection.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Expression`](expression.md), [`Anthropic Claude`](anthropic_claude.md), [`Morphological Transformation`](morphological_transformation.md), [`Text Display`](text_display.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Inner Workflow`](inner_workflow.md), [`Overlap Analysis`](overlap_analysis.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`VLM As Detector`](vlm_as_detector.md), [`Detections Consensus`](detections_consensus.md), [`Object Detection Model`](object_detection_model.md), [`Detections Filter`](detections_filter.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Detections Merge`](detections_merge.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Time in Zone`](timein_zone.md), [`SAM 3`](sam3.md), [`Size Measurement`](size_measurement.md), [`Circle Visualization`](circle_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Path Deviation`](path_deviation.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Email Notification`](email_notification.md), [`S3 Sink`](s3_sink.md), [`Camera Focus`](camera_focus.md), [`Identify Changes`](identify_changes.md), [`Byte Tracker`](byte_tracker.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SAM 3`](sam3.md), [`Image Slicer`](image_slicer.md), [`LMM For Classification`](lmm_for_classification.md), [`Dominant Color`](dominant_color.md), [`OCR Model`](ocr_model.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Switch Case`](switch_case.md), [`Google Gemma API`](google_gemma_api.md), [`OpenAI`](open_ai.md), [`Stitch Images`](stitch_images.md), [`Identify Outliers`](identify_outliers.md), [`Time in Zone`](timein_zone.md), [`Morphological Transformation`](morphological_transformation.md), [`EasyOCR`](easy_ocr.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`YOLO-World Model`](yolo_world_model.md), [`Current Time`](current_time.md), [`Blur Visualization`](blur_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Moondream2`](moondream2.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`Corner Visualization`](corner_visualization.md), [`OpenRouter`](open_router.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Pixel Color Count`](pixel_color_count.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`SAM 3`](sam3.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Google Vision OCR`](google_vision_ocr.md), [`Image Threshold`](image_threshold.md), [`Byte Tracker`](byte_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`LMM`](lmm.md), [`Polygon Visualization`](polygon_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Time in Zone`](timein_zone.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Line Counter`](line_counter.md), [`CogVLM`](cog_vlm.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Relative Static Crop`](relative_static_crop.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Grid Visualization`](grid_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Property Definition`](property_definition.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Gaze Detection`](gaze_detection.md), [`Anthropic Claude`](anthropic_claude.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Color Visualization`](color_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Detections Combine`](detections_combine.md), [`Triangle Visualization`](triangle_visualization.md), [`Data Aggregator`](data_aggregator.md), [`QR Code Generator`](qr_code_generator.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Background Subtraction`](background_subtraction.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Qwen-VL`](qwen_vl.md), [`Florence-2 Model`](florence2_model.md), [`Perspective Correction`](perspective_correction.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Cosine Similarity`](cosine_similarity.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Depth Estimation`](depth_estimation.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Contrast Equalization`](contrast_equalization.md), [`Camera Calibration`](camera_calibration.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`VLM As Detector`](vlm_as_detector.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Line Counter`](line_counter.md), [`Object Detection Model`](object_detection_model.md)

    
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


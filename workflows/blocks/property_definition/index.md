
# Property Definition



??? "Class: `PropertyDefinitionBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/formatters/property_definition/v1.py">inference.core.workflows.core_steps.formatters.property_definition.v1.PropertyDefinitionBlockV1</a>
    



Extract specific properties or fields from workflow step outputs using configurable operation chains to extract class names, confidences, counts, coordinates, OCR text, metadata, and other properties from model predictions or workflow data for data transformation, property extraction, metadata access, and value extraction workflows.

## How This Block Works

This block extracts specific properties from data by applying a chain of operations that navigate and extract values from complex data structures. The block:

1. Receives input data from any workflow step (detections, classifications, OCR results, images, or other data types)
2. Applies a chain of operations defined in the operations parameter:
   - Each operation performs a specific extraction or transformation task
   - Operations are executed sequentially, with each operation working on the result of the previous one
   - Operations can extract properties, filter data, transform formats, or combine values
3. Extracts properties based on operation type:

   **For Detection Properties:**
   - Extracts properties from object detection, instance segmentation, or keypoint detection predictions
   - Can extract: class names, confidences, counts, bounding box coordinates (x_min, y_min, x_max, y_max), centers, sizes, tracker IDs, velocities, speeds, path deviations, time in zone, polygons, and more
   - Returns lists of values (one per detection) or aggregated values

   **For Classification Properties:**
   - Extracts properties from classification predictions
   - Can extract: predicted class, confidence scores, all classes, all confidences
   - Returns single values or lists depending on the property

   **For OCR Properties:**
   - Extracts text, coordinates, and metadata from OCR results
   - Can extract: recognized text, bounding box information, confidence scores

   **For Image Properties:**
   - Extracts metadata and properties from images
   - Can extract: dimensions, format information, and other image metadata

4. Supports compound operations for complex extractions:
   - Operations can be chained to perform multi-step extractions
   - Can filter detections before extracting properties
   - Can select specific detections, transform formats, or combine multiple properties
5. Returns the extracted property value:
   - Output type depends on the property extracted (list, string, number, dictionary, etc.)
   - Returns a single output value containing the extracted property

The block uses a flexible operation system that allows extracting virtually any property from workflow data. Operations can be simple (extract a single property) or compound (filter, transform, then extract). This makes the block highly versatile for accessing specific fields from complex data structures without needing custom code.

## Common Use Cases

- **Property Extraction**: Extract specific fields from model predictions (e.g., extract class names from detections, get confidence scores, extract OCR text, get detection counts), enabling property extraction workflows
- **Metadata Access**: Access metadata and computed properties from workflow steps (e.g., extract tracker IDs, get velocity values, access time in zone, retrieve path deviations), enabling metadata access workflows
- **Data Transformation**: Transform complex data structures into simpler values for downstream use (e.g., convert detections to lists, extract coordinates, get bounding box centers, extract class lists), enabling data transformation workflows
- **Conditional Logic**: Extract values for use in conditional logic or decision making (e.g., extract counts for thresholds, get confidences for filtering, extract class names for classification, get coordinates for calculations), enabling conditional logic workflows
- **Data Formatting**: Format data for storage, display, or API responses (e.g., extract values for JSON output, format data for storage, prepare data for visualization, extract for API responses), enabling data formatting workflows
- **Analytics Extraction**: Extract metrics and measurements for analysis (e.g., extract detection counts, get confidence statistics, extract measurement values, retrieve analytics metrics), enabling analytics extraction workflows

## Connecting to Other Blocks

This block receives data from any workflow step and produces extracted property values:

- **After model blocks** (detection, classification, OCR, etc.) to extract properties from predictions (e.g., extract class names from detections, get classification results, extract OCR text), enabling model-to-property workflows
- **After analytics blocks** to extract computed metrics and measurements (e.g., extract velocity values, get time in zone, retrieve path deviations, access tracking information), enabling analytics-to-property workflows
- **Before logic blocks** like Continue If to use extracted values in conditions (e.g., continue if count exceeds threshold, filter based on extracted confidence, make decisions using extracted values), enabling property-based decision workflows
- **Before data storage blocks** to format extracted values for storage (e.g., store extracted properties, format values for logging, prepare data for storage), enabling property-to-storage workflows
- **Before visualization blocks** to provide extracted values for display (e.g., display extracted counts, show extracted text, visualize extracted metrics), enabling property visualization workflows
- **Before notification blocks** to use extracted values in notifications (e.g., include extracted counts in alerts, send extracted text in messages, use extracted values in notifications), enabling property-based notification workflows

## Requirements

This block works with any data type from workflow steps. The operations parameter defines a list of operations to perform on the input data. Each operation must be compatible with the data type and previous operation outputs. Common operations include DetectionsPropertyExtract (for detection properties), ClassificationPropertyExtract (for classification properties), and other extraction operations. The block supports compound operations (operations that can contain other operations) for complex extractions. The output type depends on the operations performed and the properties extracted - it can be a list, string, number, dictionary, or other types depending on what is extracted.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/property_definition@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `operations` | `List[Union[ClassificationPropertyExtract, ConvertDictionaryToJSON, ConvertImageToBase64, ConvertImageToJPEG, DetectionsFilter, DetectionsOffset, DetectionsPropertyExtract, DetectionsRename, DetectionsSelection, DetectionsShift, DetectionsToDictionary, Divide, ExtractDetectionProperty, ExtractFrameMetadata, ExtractImageProperty, LookupTable, Multiply, NumberRound, NumericSequenceAggregate, PickDetectionsByParentClass, RandomNumber, SequenceAggregate, SequenceApply, SequenceElementsCount, SequenceLength, SequenceMap, SortDetections, StringMatches, StringSubSequence, StringToLowerCase, StringToUpperCase, TimestampToISOFormat, ToBoolean, ToNumber, ToString]]` | List of operations to perform sequentially on the input data. Each operation performs extraction, filtering, transformation, or combination. Operations execute in order, with each operation working on the previous result. Common operations: DetectionsPropertyExtract (extract properties like class_name, confidence, count, coordinates from detections), ClassificationPropertyExtract (extract class, confidence from classifications), DetectionsFilter (filter detections before extraction), DetectionsSelection (select specific detections). Can include single or compound operations for complex extractions.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Property Definition` in version `v1`.

    - inputs: [`S3 Sink`](s3_sink.md), [`Email Notification`](email_notification.md), [`Clip Comparison`](clip_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Path Deviation`](path_deviation.md), [`SAM 3`](sam3.md), [`VLM As Detector`](vlm_as_detector.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Qwen-VL`](qwen_vl.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`YOLO-World Model`](yolo_world_model.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`OpenAI`](open_ai.md), [`VLM As Detector`](vlm_as_detector.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Email Notification`](email_notification.md), [`Seg Preview`](seg_preview.md), [`Anthropic Claude`](anthropic_claude.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Vision OCR`](google_vision_ocr.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Camera Focus`](camera_focus.md), [`Label Visualization`](label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`SAM 3`](sam3.md), [`Path Deviation`](path_deviation.md), [`Qwen3.5`](qwen3.5.md), [`Overlap Filter`](overlap_filter.md), [`Local File Sink`](local_file_sink.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SmolVLM2`](smol_vlm2.md), [`Google Gemini`](google_gemini.md), [`Rate Limiter`](rate_limiter.md), [`Motion Detection`](motion_detection.md), [`Byte Tracker`](byte_tracker.md), [`Background Color Visualization`](background_color_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Google Gemini`](google_gemini.md), [`Polygon Visualization`](polygon_visualization.md), [`Moondream2`](moondream2.md), [`Velocity`](velocity.md), [`Grid Visualization`](grid_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Florence-2 Model`](florence2_model.md), [`Delta Filter`](delta_filter.md), [`Barcode Detection`](barcode_detection.md), [`Time in Zone`](timein_zone.md), [`OCR Model`](ocr_model.md), [`Detection Event Log`](detection_event_log.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Detections Filter`](detections_filter.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Detections Merge`](detections_merge.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Detections Stabilizer`](detections_stabilizer.md), [`LMM For Classification`](lmm_for_classification.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Preprocessing`](image_preprocessing.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SIFT`](sift.md), [`Dynamic Zone`](dynamic_zone.md), [`Corner Visualization`](corner_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Halo Visualization`](halo_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Qwen3-VL`](qwen3_vl.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Time in Zone`](timein_zone.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Blur Visualization`](blur_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Property Definition`](property_definition.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Distance Measurement`](distance_measurement.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Trace Visualization`](trace_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Gaze Detection`](gaze_detection.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Dot Visualization`](dot_visualization.md), [`JSON Parser`](json_parser.md), [`Pixel Color Count`](pixel_color_count.md), [`Background Subtraction`](background_subtraction.md), [`QR Code Detection`](qr_code_detection.md), [`Text Display`](text_display.md), [`Detections Combine`](detections_combine.md), [`Bounding Rectangle`](bounding_rectangle.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Absolute Static Crop`](absolute_static_crop.md), [`CSV Formatter`](csv_formatter.md), [`Florence-2 Model`](florence2_model.md), [`Byte Tracker`](byte_tracker.md), [`Icon Visualization`](icon_visualization.md), [`Identify Outliers`](identify_outliers.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Object Detection Model`](object_detection_model.md), [`Perspective Correction`](perspective_correction.md), [`SAM 3`](sam3.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Object Detection Model`](object_detection_model.md), [`Line Counter`](line_counter.md), [`QR Code Generator`](qr_code_generator.md), [`OpenRouter`](open_router.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Image Threshold`](image_threshold.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Dynamic Crop`](dynamic_crop.md), [`Size Measurement`](size_measurement.md), [`Detections Consensus`](detections_consensus.md), [`Clip Comparison`](clip_comparison.md), [`Cache Set`](cache_set.md), [`Dominant Color`](dominant_color.md), [`Continue If`](continue_if.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Detection Offset`](detection_offset.md), [`Depth Estimation`](depth_estimation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Image Contours`](image_contours.md), [`EasyOCR`](easy_ocr.md), [`Relative Static Crop`](relative_static_crop.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Template Matching`](template_matching.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Image Blur`](image_blur.md), [`Anthropic Claude`](anthropic_claude.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Object Detection Model`](object_detection_model.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`OpenAI`](open_ai.md), [`SIFT Comparison`](sift_comparison.md), [`Slack Notification`](slack_notification.md), [`Image Stack`](image_stack.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Stitch Images`](stitch_images.md), [`Buffer`](buffer.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Image Slicer`](image_slicer.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Slicer`](image_slicer.md), [`Cosine Similarity`](cosine_similarity.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Cache Get`](cache_get.md), [`LMM`](lmm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Expression`](expression.md), [`Detections Transformation`](detections_transformation.md), [`Color Visualization`](color_visualization.md), [`Google Gemini`](google_gemini.md), [`Data Aggregator`](data_aggregator.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Camera Focus`](camera_focus.md), [`Camera Calibration`](camera_calibration.md), [`Detections Stitch`](detections_stitch.md), [`Byte Tracker`](byte_tracker.md), [`Ellipse Visualization`](ellipse_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Identify Changes`](identify_changes.md), [`SORT Tracker`](sort_tracker.md), [`Mask Visualization`](mask_visualization.md), [`GLM-OCR`](glmocr.md), [`Crop Visualization`](crop_visualization.md), [`Circle Visualization`](circle_visualization.md), [`CogVLM`](cog_vlm.md), [`Inner Workflow`](inner_workflow.md), [`Dimension Collapse`](dimension_collapse.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Contrast Equalization`](contrast_equalization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Webhook Sink`](webhook_sink.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemma`](google_gemma.md)
    - outputs: [`S3 Sink`](s3_sink.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Email Notification`](email_notification.md), [`Morphological Transformation`](morphological_transformation.md), [`SAM 3`](sam3.md), [`Qwen-VL`](qwen_vl.md), [`Path Deviation`](path_deviation.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Detector`](vlm_as_detector.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`YOLO-World Model`](yolo_world_model.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`OpenAI`](open_ai.md), [`VLM As Detector`](vlm_as_detector.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Email Notification`](email_notification.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Seg Preview`](seg_preview.md), [`Anthropic Claude`](anthropic_claude.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Google Vision OCR`](google_vision_ocr.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`SAM 3`](sam3.md), [`Label Visualization`](label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Camera Focus`](camera_focus.md), [`Path Deviation`](path_deviation.md), [`Qwen3.5`](qwen3.5.md), [`Overlap Filter`](overlap_filter.md), [`Local File Sink`](local_file_sink.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SmolVLM2`](smol_vlm2.md), [`Google Gemini`](google_gemini.md), [`Rate Limiter`](rate_limiter.md), [`Motion Detection`](motion_detection.md), [`Byte Tracker`](byte_tracker.md), [`Background Color Visualization`](background_color_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Google Gemini`](google_gemini.md), [`Polygon Visualization`](polygon_visualization.md), [`Moondream2`](moondream2.md), [`Velocity`](velocity.md), [`SIFT Comparison`](sift_comparison.md), [`Detection Event Log`](detection_event_log.md), [`Florence-2 Model`](florence2_model.md), [`Delta Filter`](delta_filter.md), [`Grid Visualization`](grid_visualization.md), [`Time in Zone`](timein_zone.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OCR Model`](ocr_model.md), [`Barcode Detection`](barcode_detection.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Detections Filter`](detections_filter.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Detections Merge`](detections_merge.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Detections Stabilizer`](detections_stabilizer.md), [`LMM For Classification`](lmm_for_classification.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Preprocessing`](image_preprocessing.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SIFT`](sift.md), [`Dynamic Zone`](dynamic_zone.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Corner Visualization`](corner_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Halo Visualization`](halo_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Qwen3-VL`](qwen3_vl.md), [`Time in Zone`](timein_zone.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Blur Visualization`](blur_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Property Definition`](property_definition.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Distance Measurement`](distance_measurement.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Trace Visualization`](trace_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Gaze Detection`](gaze_detection.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Dot Visualization`](dot_visualization.md), [`JSON Parser`](json_parser.md), [`Pixel Color Count`](pixel_color_count.md), [`QR Code Detection`](qr_code_detection.md), [`Background Subtraction`](background_subtraction.md), [`Text Display`](text_display.md), [`Detections Combine`](detections_combine.md), [`Bounding Rectangle`](bounding_rectangle.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Absolute Static Crop`](absolute_static_crop.md), [`CSV Formatter`](csv_formatter.md), [`Florence-2 Model`](florence2_model.md), [`Byte Tracker`](byte_tracker.md), [`Icon Visualization`](icon_visualization.md), [`Identify Outliers`](identify_outliers.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Object Detection Model`](object_detection_model.md), [`Perspective Correction`](perspective_correction.md), [`SAM 3`](sam3.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Object Detection Model`](object_detection_model.md), [`Line Counter`](line_counter.md), [`QR Code Generator`](qr_code_generator.md), [`OpenRouter`](open_router.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Image Threshold`](image_threshold.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Dynamic Crop`](dynamic_crop.md), [`Detections Consensus`](detections_consensus.md), [`Size Measurement`](size_measurement.md), [`Clip Comparison`](clip_comparison.md), [`Cache Set`](cache_set.md), [`Dominant Color`](dominant_color.md), [`Continue If`](continue_if.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Detection Offset`](detection_offset.md), [`Depth Estimation`](depth_estimation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Contours`](image_contours.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`EasyOCR`](easy_ocr.md), [`Relative Static Crop`](relative_static_crop.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Template Matching`](template_matching.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Image Blur`](image_blur.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Anthropic Claude`](anthropic_claude.md), [`Triangle Visualization`](triangle_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`SIFT Comparison`](sift_comparison.md), [`OpenAI`](open_ai.md), [`Slack Notification`](slack_notification.md), [`Image Stack`](image_stack.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Stitch Images`](stitch_images.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Buffer`](buffer.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Slicer`](image_slicer.md), [`Cosine Similarity`](cosine_similarity.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Cache Get`](cache_get.md), [`LMM`](lmm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Data Aggregator`](data_aggregator.md), [`Detections Transformation`](detections_transformation.md), [`Color Visualization`](color_visualization.md), [`Google Gemini`](google_gemini.md), [`Expression`](expression.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Camera Focus`](camera_focus.md), [`Camera Calibration`](camera_calibration.md), [`Detections Stitch`](detections_stitch.md), [`Byte Tracker`](byte_tracker.md), [`Ellipse Visualization`](ellipse_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Identify Changes`](identify_changes.md), [`SORT Tracker`](sort_tracker.md), [`Mask Visualization`](mask_visualization.md), [`GLM-OCR`](glmocr.md), [`Crop Visualization`](crop_visualization.md), [`Circle Visualization`](circle_visualization.md), [`CogVLM`](cog_vlm.md), [`Inner Workflow`](inner_workflow.md), [`Dimension Collapse`](dimension_collapse.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Contrast Equalization`](contrast_equalization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Webhook Sink`](webhook_sink.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemma`](google_gemma.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Property Definition` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `data` (*[`*`](../kinds/wildcard.md)*): Input data from any workflow step to extract properties from. Can be detections, classifications, OCR results, images, or any other workflow output. The data type determines which operations are applicable. Examples: detection predictions for extracting class names, classification results for extracting predicted class, OCR results for extracting text..

    - output
    
        - `output` ([`*`](../kinds/wildcard.md)): Equivalent of any element.



??? tip "Example JSON definition of step `Property Definition` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/property_definition@v1",
	    "data": "$steps.object_detection_model.predictions",
	    "operations": [
	        {
	            "property_name": "class_name",
	            "type": "DetectionsPropertyExtract"
	        }
	    ]
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    


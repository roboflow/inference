
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

    - inputs: [`Triangle Visualization`](triangle_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Florence-2 Model`](florence2_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Image Stack`](image_stack.md), [`Path Deviation`](path_deviation.md), [`Distance Measurement`](distance_measurement.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`Dominant Color`](dominant_color.md), [`Background Subtraction`](background_subtraction.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Camera Focus`](camera_focus.md), [`SmolVLM2`](smol_vlm2.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Google Gemini`](google_gemini.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Cache Get`](cache_get.md), [`Email Notification`](email_notification.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Inner Workflow`](inner_workflow.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Email Notification`](email_notification.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Contours`](image_contours.md), [`Icon Visualization`](icon_visualization.md), [`Overlap Filter`](overlap_filter.md), [`Circle Visualization`](circle_visualization.md), [`Buffer`](buffer.md), [`Mask Visualization`](mask_visualization.md), [`OpenAI`](open_ai.md), [`Reference Path Visualization`](reference_path_visualization.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Velocity`](velocity.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Time in Zone`](timein_zone.md), [`Image Slicer`](image_slicer.md), [`Qwen3.5`](qwen3.5.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Polygon Visualization`](polygon_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`JSON Parser`](json_parser.md), [`SAM 3`](sam3.md), [`QR Code Detection`](qr_code_detection.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Detections Stitch`](detections_stitch.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Grid Visualization`](grid_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Color Visualization`](color_visualization.md), [`Time in Zone`](timein_zone.md), [`VLM As Detector`](vlm_as_detector.md), [`Dot Visualization`](dot_visualization.md), [`Size Measurement`](size_measurement.md), [`OCR Model`](ocr_model.md), [`Qwen3-VL`](qwen3_vl.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Trace Visualization`](trace_visualization.md), [`Slack Notification`](slack_notification.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Moondream2`](moondream2.md), [`Byte Tracker`](byte_tracker.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Template Matching`](template_matching.md), [`Google Gemma API`](google_gemma_api.md), [`Anthropic Claude`](anthropic_claude.md), [`LMM For Classification`](lmm_for_classification.md), [`VLM As Classifier`](vlm_as_classifier.md), [`OpenAI`](open_ai.md), [`Byte Tracker`](byte_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Webhook Sink`](webhook_sink.md), [`Mask Edge Snap`](mask_edge_snap.md), [`GLM-OCR`](glmocr.md), [`Background Color Visualization`](background_color_visualization.md), [`Time in Zone`](timein_zone.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Detections Transformation`](detections_transformation.md), [`SORT Tracker`](sort_tracker.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Stitch Images`](stitch_images.md), [`S3 Sink`](s3_sink.md), [`Google Vision OCR`](google_vision_ocr.md), [`Detections Combine`](detections_combine.md), [`Image Blur`](image_blur.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Byte Tracker`](byte_tracker.md), [`Morphological Transformation`](morphological_transformation.md), [`Rate Limiter`](rate_limiter.md), [`Cosine Similarity`](cosine_similarity.md), [`Identify Changes`](identify_changes.md), [`QR Code Generator`](qr_code_generator.md), [`Image Preprocessing`](image_preprocessing.md), [`Detection Event Log`](detection_event_log.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Barcode Detection`](barcode_detection.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Crop Visualization`](crop_visualization.md), [`Line Counter`](line_counter.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Delta Filter`](delta_filter.md), [`Relative Static Crop`](relative_static_crop.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Identify Outliers`](identify_outliers.md), [`Perspective Correction`](perspective_correction.md), [`Halo Visualization`](halo_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Seg Preview`](seg_preview.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Continue If`](continue_if.md), [`Line Counter`](line_counter.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Text Display`](text_display.md), [`SIFT`](sift.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Gaze Detection`](gaze_detection.md), [`Data Aggregator`](data_aggregator.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Camera Focus`](camera_focus.md), [`Anthropic Claude`](anthropic_claude.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Motion Detection`](motion_detection.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Property Definition`](property_definition.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Image Threshold`](image_threshold.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`Blur Visualization`](blur_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Corner Visualization`](corner_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Label Visualization`](label_visualization.md), [`Detections Consensus`](detections_consensus.md), [`CSV Formatter`](csv_formatter.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detections Merge`](detections_merge.md), [`Detection Offset`](detection_offset.md), [`Camera Calibration`](camera_calibration.md), [`SAM 3`](sam3.md), [`Environment Secrets Store`](environment_secrets_store.md), [`EasyOCR`](easy_ocr.md), [`Morphological Transformation`](morphological_transformation.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Path Deviation`](path_deviation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`CogVLM`](cog_vlm.md), [`LMM`](lmm.md), [`Depth Estimation`](depth_estimation.md), [`SIFT Comparison`](sift_comparison.md), [`Detections Filter`](detections_filter.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Expression`](expression.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Pixel Color Count`](pixel_color_count.md), [`Local File Sink`](local_file_sink.md), [`Cache Set`](cache_set.md), [`Bounding Rectangle`](bounding_rectangle.md)
    - outputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`Triangle Visualization`](triangle_visualization.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Florence-2 Model`](florence2_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Path Deviation`](path_deviation.md), [`Distance Measurement`](distance_measurement.md), [`Image Stack`](image_stack.md), [`OpenAI`](open_ai.md), [`Dominant Color`](dominant_color.md), [`Clip Comparison`](clip_comparison.md), [`Background Subtraction`](background_subtraction.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Camera Focus`](camera_focus.md), [`SmolVLM2`](smol_vlm2.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Google Gemini`](google_gemini.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Cache Get`](cache_get.md), [`Email Notification`](email_notification.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Inner Workflow`](inner_workflow.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Email Notification`](email_notification.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Contours`](image_contours.md), [`Icon Visualization`](icon_visualization.md), [`Overlap Filter`](overlap_filter.md), [`Circle Visualization`](circle_visualization.md), [`Mask Visualization`](mask_visualization.md), [`OpenAI`](open_ai.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Velocity`](velocity.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Buffer`](buffer.md), [`Time in Zone`](timein_zone.md), [`Image Slicer`](image_slicer.md), [`Qwen3.5`](qwen3.5.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Polygon Visualization`](polygon_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`SAM 3`](sam3.md), [`JSON Parser`](json_parser.md), [`QR Code Detection`](qr_code_detection.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Detections Stitch`](detections_stitch.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Grid Visualization`](grid_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Color Visualization`](color_visualization.md), [`Cache Set`](cache_set.md), [`Time in Zone`](timein_zone.md), [`VLM As Detector`](vlm_as_detector.md), [`Dot Visualization`](dot_visualization.md), [`Size Measurement`](size_measurement.md), [`OCR Model`](ocr_model.md), [`Qwen3-VL`](qwen3_vl.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Trace Visualization`](trace_visualization.md), [`Slack Notification`](slack_notification.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Moondream2`](moondream2.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Byte Tracker`](byte_tracker.md), [`Template Matching`](template_matching.md), [`Google Gemma API`](google_gemma_api.md), [`LMM For Classification`](lmm_for_classification.md), [`Anthropic Claude`](anthropic_claude.md), [`VLM As Classifier`](vlm_as_classifier.md), [`OpenAI`](open_ai.md), [`Byte Tracker`](byte_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Webhook Sink`](webhook_sink.md), [`GLM-OCR`](glmocr.md), [`Background Color Visualization`](background_color_visualization.md), [`Time in Zone`](timein_zone.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Detections Transformation`](detections_transformation.md), [`SORT Tracker`](sort_tracker.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Stitch Images`](stitch_images.md), [`S3 Sink`](s3_sink.md), [`Detections Combine`](detections_combine.md), [`Google Vision OCR`](google_vision_ocr.md), [`Image Blur`](image_blur.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Byte Tracker`](byte_tracker.md), [`Morphological Transformation`](morphological_transformation.md), [`Identify Changes`](identify_changes.md), [`Cosine Similarity`](cosine_similarity.md), [`QR Code Generator`](qr_code_generator.md), [`Rate Limiter`](rate_limiter.md), [`Image Preprocessing`](image_preprocessing.md), [`Detection Event Log`](detection_event_log.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Barcode Detection`](barcode_detection.md), [`Line Counter`](line_counter.md), [`Crop Visualization`](crop_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Delta Filter`](delta_filter.md), [`Relative Static Crop`](relative_static_crop.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Identify Outliers`](identify_outliers.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Perspective Correction`](perspective_correction.md), [`Halo Visualization`](halo_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Object Detection Model`](object_detection_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Seg Preview`](seg_preview.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Continue If`](continue_if.md), [`Line Counter`](line_counter.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Text Display`](text_display.md), [`SIFT`](sift.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Gaze Detection`](gaze_detection.md), [`Data Aggregator`](data_aggregator.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Camera Focus`](camera_focus.md), [`Anthropic Claude`](anthropic_claude.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Motion Detection`](motion_detection.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Property Definition`](property_definition.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Image Threshold`](image_threshold.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Dimension Collapse`](dimension_collapse.md), [`Blur Visualization`](blur_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Label Visualization`](label_visualization.md), [`Detections Consensus`](detections_consensus.md), [`CSV Formatter`](csv_formatter.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detections Merge`](detections_merge.md), [`Detection Offset`](detection_offset.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Calibration`](camera_calibration.md), [`SAM 3`](sam3.md), [`EasyOCR`](easy_ocr.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Path Deviation`](path_deviation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`SIFT Comparison`](sift_comparison.md), [`LMM`](lmm.md), [`Depth Estimation`](depth_estimation.md), [`Anthropic Claude`](anthropic_claude.md), [`Detections Filter`](detections_filter.md), [`CogVLM`](cog_vlm.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Expression`](expression.md), [`Pixel Color Count`](pixel_color_count.md), [`Local File Sink`](local_file_sink.md), [`Line Counter Visualization`](line_counter_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md)

    
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


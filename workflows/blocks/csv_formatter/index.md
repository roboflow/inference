
# CSV Formatter



??? "Class: `CSVFormatterBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/formatters/csv/v1.py">inference.core.workflows.core_steps.formatters.csv.v1.CSVFormatterBlockV1</a>
    



Convert workflow data into structured CSV format by defining custom columns, applying data transformations, and aggregating batch data into CSV documents with automatic timestamp tracking for logging, reporting, and data export workflows.

## How This Block Works

This block formats workflow data into CSV (Comma-Separated Values) format by organizing data from multiple sources into structured columns. The block:

1. Takes data references from `columns_data` dictionary that maps column names to workflow data sources (selectors, static values, or workflow inputs)
2. Optionally applies data transformation operations using `columns_operations`, which uses the Query Language (UQL) to transform column data (e.g., extract properties from detections, perform calculations, format values)
3. Automatically adds a `timestamp` column with the current UTC time in ISO format (e.g., `2024-10-18T14:09:57.622297+00:00`) to each row - note that "timestamp" is a reserved column name
4. Handles batch inputs by aggregating multiple data points into rows:
   - For single input (`batch_size=1`): Creates CSV with header row and one data row
   - For batch inputs (`batch_size>1`): Creates CSV with header row and one row per input, aggregating all rows into a single CSV document that is output only in the last batch element (earlier elements return empty CSV content)
5. Aligns batch parameters when multiple batch inputs are provided, broadcasting non-batch parameters to match the maximum batch size
6. Converts the structured data dictionary into CSV format using pandas DataFrame serialization
7. Returns `csv_content` as a string containing the complete CSV document (header and data rows)

The block supports flexible column definition where each column can reference different workflow data sources (detection predictions, classification results, workflow inputs, computed values, etc.) and optionally apply transformations to extract specific properties or format data. The automatic timestamp column enables temporal tracking of when each CSV row was generated, useful for logging and time-series data collection. Batch aggregation allows the block to collect data from multiple workflow executions and combine them into a single CSV document, which is particularly useful for batch processing workflows where you want to log multiple detections, images, or analysis results into one CSV file.

## Common Use Cases

- **Detection Logging and Reporting**: Create CSV logs of detection results (e.g., log class names, confidence scores, bounding box coordinates from object detection models), enabling structured logging of inference results for analysis, debugging, or audit trails
- **Time-Series Data Collection**: Aggregate workflow metrics, counts, or analysis results over time into CSV format (e.g., log line counter counts, zone occupancy, detection frequencies), creating time-stamped datasets for trend analysis or reporting
- **Batch Data Export**: Collect and aggregate data from batch processing workflows into CSV files (e.g., export all detections from a batch of images, collect metrics from multiple workflow runs), enabling efficient bulk data export and reporting
- **Structured Data Transformation**: Extract and format specific properties from complex workflow outputs (e.g., extract class names from detections, convert nested data structures into flat CSV columns), enabling data transformation for downstream analysis or external systems
- **Integration with External Systems**: Format workflow data for compatibility with external tools (e.g., create CSV files for spreadsheet analysis, database import, or business intelligence tools), enabling seamless data export and integration workflows
- **Data Aggregation and Analysis**: Combine data from multiple workflow sources into structured CSV format (e.g., merge detection results with metadata, combine model outputs with reference data), enabling comprehensive data collection and analysis workflows

## Connecting to Other Blocks

The CSV content from this block can be connected to:

- **Detection or analysis blocks** (e.g., Object Detection Model, Instance Segmentation Model, Classification Model, Keypoint Detection Model, Line Counter, Time in Zone) to format their outputs into CSV columns, enabling structured logging and export of inference results and analytics data
- **Data storage blocks** (e.g., Local File Sink) to save CSV files to disk, enabling persistent storage of formatted workflow data for later analysis or reporting
- **Notification blocks** (e.g., Email Notification, Slack Notification) to attach or include CSV content in notifications, enabling CSV reports to be sent as email attachments or included in message bodies
- **Webhook blocks** (e.g., Webhook Sink) to send CSV content to external APIs or services, enabling integration with external systems that consume CSV data
- **Other formatter blocks** (e.g., JSON Parser, Expression) to further process CSV content or convert it to other formats, enabling multi-stage data transformation workflows
- **Batch processing workflows** where multiple data points need to be aggregated into a single CSV document, allowing comprehensive logging and export of batch processing results


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/csv_formatter@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `columns_data` | `Dict[str, Union[bool, float, int, str]]` | Dictionary mapping column names to data sources for constructing CSV columns. Keys are column names (note: 'timestamp' is reserved and cannot be used). Values can be selectors referencing workflow data (e.g., '$steps.model.predictions', '$inputs.data'), static values (strings, numbers, booleans), or a mix of both. Each key-value pair creates one CSV column. Supports batch inputs - if values are batches, the CSV will aggregate all batch elements into rows. Example: {'predictions': '$steps.object_detection.predictions', 'count': '$steps.line_counter.count_in'} creates CSV columns named 'predictions' and 'count'.. | ✅ |
| `columns_operations` | `Dict[str, List[Union[ClassificationPropertyExtract, ConvertDictionaryToJSON, ConvertImageToBase64, ConvertImageToJPEG, DetectionsFilter, DetectionsOffset, DetectionsPropertyExtract, DetectionsRename, DetectionsSelection, DetectionsShift, DetectionsToDictionary, Divide, ExtractDetectionProperty, ExtractFrameMetadata, ExtractImageProperty, LookupTable, Multiply, NumberRound, NumericSequenceAggregate, PickDetectionsByParentClass, RandomNumber, SequenceAggregate, SequenceApply, SequenceElementsCount, SequenceLength, SequenceMap, SortDetections, StringMatches, StringSubSequence, StringToLowerCase, StringToUpperCase, TimestampToISOFormat, ToBoolean, ToNumber, ToString]]]` | Optional dictionary mapping column names to Query Language (UQL) operation definitions for transforming column data before CSV formatting. Keys must match column names defined in columns_data. Values are lists of UQL operations (e.g., DetectionsPropertyExtract to extract class names from detections, string operations, calculations) that transform the raw column data. Operations are applied in sequence to each column's data. If a column name is not in this dictionary, the data is used as-is without transformation. Example: {'predictions': [{'type': 'DetectionsPropertyExtract', 'property_name': 'class_name'}]} extracts class names from detection predictions.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `CSV Formatter` in version `v1`.

    - inputs: [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`OCR Model`](ocr_model.md), [`Gaze Detection`](gaze_detection.md), [`Image Slicer`](image_slicer.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Distance Measurement`](distance_measurement.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Color Visualization`](color_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Data Aggregator`](data_aggregator.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Byte Tracker`](byte_tracker.md), [`Detections Consensus`](detections_consensus.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Barcode Detection`](barcode_detection.md), [`Cache Set`](cache_set.md), [`Webhook Sink`](webhook_sink.md), [`Continue If`](continue_if.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Trace Visualization`](trace_visualization.md), [`Camera Focus`](camera_focus.md), [`Object Detection Model`](object_detection_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`OpenAI`](open_ai.md), [`Buffer`](buffer.md), [`SAM 3`](sam3.md), [`Size Measurement`](size_measurement.md), [`Image Threshold`](image_threshold.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Florence-2 Model`](florence2_model.md), [`Halo Visualization`](halo_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Path Deviation`](path_deviation.md), [`GLM-OCR`](glmocr.md), [`Dot Visualization`](dot_visualization.md), [`S3 Sink`](s3_sink.md), [`Path Deviation`](path_deviation.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Seg Preview`](seg_preview.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Dynamic Zone`](dynamic_zone.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Line Counter`](line_counter.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Text Display`](text_display.md), [`CSV Formatter`](csv_formatter.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Detections Merge`](detections_merge.md), [`Perspective Correction`](perspective_correction.md), [`Anthropic Claude`](anthropic_claude.md), [`Line Counter`](line_counter.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Velocity`](velocity.md), [`Depth Estimation`](depth_estimation.md), [`Overlap Filter`](overlap_filter.md), [`Rate Limiter`](rate_limiter.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Polygon Visualization`](polygon_visualization.md), [`SmolVLM2`](smol_vlm2.md), [`SIFT`](sift.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemini`](google_gemini.md), [`Label Visualization`](label_visualization.md), [`Expression`](expression.md), [`Grid Visualization`](grid_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Contrast Equalization`](contrast_equalization.md), [`Triangle Visualization`](triangle_visualization.md), [`Property Definition`](property_definition.md), [`Halo Visualization`](halo_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`Dominant Color`](dominant_color.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`Slack Notification`](slack_notification.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Detections Stitch`](detections_stitch.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Object Detection Model`](object_detection_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Email Notification`](email_notification.md), [`Google Gemma API`](google_gemma_api.md), [`Google Vision OCR`](google_vision_ocr.md), [`Identify Outliers`](identify_outliers.md), [`Google Gemini`](google_gemini.md), [`Image Preprocessing`](image_preprocessing.md), [`EasyOCR`](easy_ocr.md), [`Detections Combine`](detections_combine.md), [`Object Detection Model`](object_detection_model.md), [`Cosine Similarity`](cosine_similarity.md), [`OpenAI`](open_ai.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Byte Tracker`](byte_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Time in Zone`](timein_zone.md), [`Qwen3-VL`](qwen3_vl.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`YOLO-World Model`](yolo_world_model.md), [`Inner Workflow`](inner_workflow.md), [`Detection Offset`](detection_offset.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Template Matching`](template_matching.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Stitch Images`](stitch_images.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`CogVLM`](cog_vlm.md), [`Crop Visualization`](crop_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Florence-2 Model`](florence2_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Time in Zone`](timein_zone.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`SAM 3`](sam3.md), [`QR Code Detection`](qr_code_detection.md), [`Icon Visualization`](icon_visualization.md), [`Detections Filter`](detections_filter.md), [`Local File Sink`](local_file_sink.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Image Contours`](image_contours.md), [`JSON Parser`](json_parser.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Time in Zone`](timein_zone.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`Anthropic Claude`](anthropic_claude.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Detector`](vlm_as_detector.md), [`LMM`](lmm.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Pixel Color Count`](pixel_color_count.md), [`Identify Changes`](identify_changes.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Image Blur`](image_blur.md), [`Image Slicer`](image_slicer.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`SAM 3`](sam3.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Corner Visualization`](corner_visualization.md), [`Delta Filter`](delta_filter.md), [`Dynamic Crop`](dynamic_crop.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Keypoint Visualization`](keypoint_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Moondream2`](moondream2.md), [`QR Code Generator`](qr_code_generator.md), [`Camera Focus`](camera_focus.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Background Color Visualization`](background_color_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Cache Get`](cache_get.md)
    - outputs: [`CLIP Embedding Model`](clip_embedding_model.md), [`Detections Stitch`](detections_stitch.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Email Notification`](email_notification.md), [`Google Gemma API`](google_gemma_api.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Google Vision OCR`](google_vision_ocr.md), [`Distance Measurement`](distance_measurement.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`Image Preprocessing`](image_preprocessing.md), [`Color Visualization`](color_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI`](open_ai.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Time in Zone`](timein_zone.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Cache Set`](cache_set.md), [`Webhook Sink`](webhook_sink.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`YOLO-World Model`](yolo_world_model.md), [`OpenAI`](open_ai.md), [`SAM 3`](sam3.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Size Measurement`](size_measurement.md), [`Image Threshold`](image_threshold.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Florence-2 Model`](florence2_model.md), [`Halo Visualization`](halo_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`CogVLM`](cog_vlm.md), [`Crop Visualization`](crop_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Path Deviation`](path_deviation.md), [`Time in Zone`](timein_zone.md), [`GLM-OCR`](glmocr.md), [`Dot Visualization`](dot_visualization.md), [`S3 Sink`](s3_sink.md), [`Path Deviation`](path_deviation.md), [`SAM 3`](sam3.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Seg Preview`](seg_preview.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Local File Sink`](local_file_sink.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Icon Visualization`](icon_visualization.md), [`Line Counter`](line_counter.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Time in Zone`](timein_zone.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Text Display`](text_display.md), [`Clip Comparison`](clip_comparison.md), [`LMM`](lmm.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Perspective Correction`](perspective_correction.md), [`Anthropic Claude`](anthropic_claude.md), [`Line Counter`](line_counter.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Depth Estimation`](depth_estimation.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Image Blur`](image_blur.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Polygon Visualization`](polygon_visualization.md), [`SAM 3`](sam3.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`OpenAI`](open_ai.md), [`Google Gemini`](google_gemini.md), [`Label Visualization`](label_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Contrast Equalization`](contrast_equalization.md), [`Dynamic Crop`](dynamic_crop.md), [`Triangle Visualization`](triangle_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Moondream2`](moondream2.md), [`QR Code Generator`](qr_code_generator.md), [`Halo Visualization`](halo_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Background Color Visualization`](background_color_visualization.md), [`Email Notification`](email_notification.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Slack Notification`](slack_notification.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Cache Get`](cache_get.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`CSV Formatter` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `columns_data` (*[`*`](../kinds/wildcard.md)*): Dictionary mapping column names to data sources for constructing CSV columns. Keys are column names (note: 'timestamp' is reserved and cannot be used). Values can be selectors referencing workflow data (e.g., '$steps.model.predictions', '$inputs.data'), static values (strings, numbers, booleans), or a mix of both. Each key-value pair creates one CSV column. Supports batch inputs - if values are batches, the CSV will aggregate all batch elements into rows. Example: {'predictions': '$steps.object_detection.predictions', 'count': '$steps.line_counter.count_in'} creates CSV columns named 'predictions' and 'count'..

    - output
    
        - `csv_content` ([`string`](../kinds/string.md)): String value.



??? tip "Example JSON definition of step `CSV Formatter` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/csv_formatter@v1",
	    "columns_data": {
	        "predictions": "$steps.model.predictions",
	        "reference": "$inputs.reference_class_names"
	    },
	    "columns_operations": {
	        "predictions": [
	            {
	                "property_name": "class_name",
	                "type": "DetectionsPropertyExtract"
	            }
	        ]
	    }
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    


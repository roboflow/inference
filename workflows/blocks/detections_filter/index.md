
# Detections Filter



??? "Class: `DetectionsFilterBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/transformations/detections_filter/v1.py">inference.core.workflows.core_steps.transformations.detections_filter.v1.DetectionsFilterBlockV1</a>
    



Filter detection predictions based on customizable conditions, selectively removing detections that don't meet specified criteria (e.g., class names, confidence scores, bounding box properties) while preserving only the detections that match your filtering logic.

## How This Block Works

This block applies conditional filtering to detection predictions using a flexible query language system. The block:

1. Takes detection predictions (object detection, instance segmentation, or keypoint detection) and filtering operation definitions as input
2. Evaluates each detection against the filtering conditions specified in the `operations` parameter
3. Extracts detection properties (e.g., class_name, confidence, bounding box coordinates) using property extraction operations
4. Compares extracted properties against criteria using binary statements (e.g., class_name in list, confidence > threshold)
5. Filters out detections that don't match the conditions, keeping only detections that satisfy the filter criteria
6. Returns filtered predictions containing only the detections that passed the filter conditions

The block uses a query language system that supports extracting various detection properties (class names, confidence scores, bounding box coordinates, etc.) and applying conditional logic to filter detections. Filtering operations can check if properties are in lists, compare numeric values, check string equality, or use other comparators. The `operations_parameters` dictionary provides runtime values (like class name lists or thresholds) that are referenced in the filtering operations, allowing dynamic filtering criteria that can change based on workflow inputs or computed values. Multiple filtering operations can be chained together to create complex filtering logic.

## Common Use Cases

- **Class-Based Filtering**: Filter detections to keep only specific object classes (e.g., keep only "person" and "car" detections, remove all others), enabling focused processing on relevant object types while excluding unwanted detections
- **Confidence Threshold Filtering**: Remove low-confidence detections to improve detection quality (e.g., keep detections with confidence > 0.7, filter out uncertain predictions), ensuring downstream processing works with reliable detections
- **Multi-Criteria Filtering**: Apply multiple filtering conditions simultaneously (e.g., keep detections where class_name is in allowed list AND confidence > threshold), combining class and confidence filtering for precise control
- **Dynamic Filtering Based on Workflow State**: Use workflow inputs or computed values to determine filtering criteria (e.g., filter classes based on user input, adjust confidence threshold based on lighting conditions), enabling adaptive filtering that responds to changing conditions
- **Pre-Processing for Downstream Blocks**: Filter detections before passing to visualization, counting, or storage blocks (e.g., remove false positives before counting, filter out background classes before visualization), reducing noise and improving accuracy of subsequent operations
- **Selective Processing Workflows**: Route different filtered subsets to different downstream blocks (e.g., filter high-confidence detections to one path, low-confidence to another), enabling conditional processing based on detection quality or type

## Connecting to Other Blocks

The filtered predictions from this block can be connected to:

- **Detection model blocks** (e.g., Object Detection Model, Instance Segmentation Model, Keypoint Detection Model) to receive predictions that are filtered based on class, confidence, or other properties
- **Visualization blocks** (e.g., Bounding Box Visualization, Polygon Visualization, Label Visualization) to display only the filtered detections, reducing visual clutter and focusing on relevant objects
- **Counting and analytics blocks** (e.g., Line Counter, Time in Zone, Velocity) to count or analyze only specific filtered classes or confidence levels, ensuring accurate metrics for the objects of interest
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload, Webhook Sink) to save or transmit only filtered detection results, reducing storage and bandwidth usage by excluding irrelevant detections
- **Other transformation blocks** (e.g., Detections Merge, Detections Transform, Detection Offset) to apply additional transformations to the filtered subset, enabling complex processing pipelines on filtered detections
- **Flow control blocks** (e.g., Continue If, Rate Limiter) to conditionally trigger downstream processing based on whether filtered detections meet certain criteria, enabling conditional workflows based on filtered results


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/detections_filter@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `operations` | `List[Union[ClassificationPropertyExtract, ConvertDictionaryToJSON, ConvertImageToBase64, ConvertImageToJPEG, DetectionsFilter, DetectionsOffset, DetectionsPropertyExtract, DetectionsRename, DetectionsSelection, DetectionsShift, DetectionsToDictionary, Divide, ExtractDetectionProperty, ExtractFrameMetadata, ExtractImageProperty, LookupTable, Multiply, NumberRound, NumericSequenceAggregate, PickDetectionsByParentClass, RandomNumber, SequenceAggregate, SequenceApply, SequenceElementsCount, SequenceLength, SequenceMap, SortDetections, StringMatches, StringSubSequence, StringToLowerCase, StringToUpperCase, TimestampToISOFormat, ToBoolean, ToNumber, ToString]]` | Definition of filtering logic using the query language system. Specifies one or more filtering operations (e.g., DetectionsFilter) that use StatementGroup syntax to define conditional logic. Each operation can extract detection properties (class_name, confidence, coordinates, etc.) and compare them using binary statements (e.g., class_name in list, confidence > threshold). Multiple operations can be chained to create complex filtering logic. The operations reference parameter names from operations_parameters to access runtime values.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Detections Filter` in version `v1`.

    - inputs: [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Object Detection Model`](object_detection_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`PLC Writer`](plc_writer.md), [`Distance Measurement`](distance_measurement.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Contrast Equalization`](contrast_equalization.md), [`Google Gemini`](google_gemini.md), [`Background Subtraction`](background_subtraction.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Detections Filter`](detections_filter.md), [`Detections Merge`](detections_merge.md), [`VLM As Detector`](vlm_as_detector.md), [`Florence-2 Model`](florence2_model.md), [`Dynamic Crop`](dynamic_crop.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Halo Visualization`](halo_visualization.md), [`Seg Preview`](seg_preview.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Florence-2 Model`](florence2_model.md), [`Line Counter`](line_counter.md), [`Cache Set`](cache_set.md), [`OpenAI`](open_ai.md), [`Data Aggregator`](data_aggregator.md), [`Time in Zone`](timein_zone.md), [`Google Gemma`](google_gemma.md), [`Byte Tracker`](byte_tracker.md), [`Buffer`](buffer.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Email Notification`](email_notification.md), [`Detections Consensus`](detections_consensus.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Crop Visualization`](crop_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Line Counter`](line_counter.md), [`Detections Combine`](detections_combine.md), [`Object Detection Model`](object_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Google Gemini`](google_gemini.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Qwen3.5`](qwen3.5.md), [`Delta Filter`](delta_filter.md), [`Object Detection Model`](object_detection_model.md), [`Time in Zone`](timein_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Expression`](expression.md), [`PP-OCR`](ppocr.md), [`Blur Visualization`](blur_visualization.md), [`EasyOCR`](easy_ocr.md), [`Clip Comparison`](clip_comparison.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`CSV Formatter`](csv_formatter.md), [`Color Visualization`](color_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Motion Detection`](motion_detection.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Rate Limiter`](rate_limiter.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Current Time`](current_time.md), [`Trace Visualization`](trace_visualization.md), [`Cache Get`](cache_get.md), [`PLC Reader`](plc_reader.md), [`QR Code Generator`](qr_code_generator.md), [`Google Gemini`](google_gemini.md), [`Detections Stitch`](detections_stitch.md), [`Image Stack`](image_stack.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Continue If`](continue_if.md), [`Webhook Sink`](webhook_sink.md), [`Cosine Similarity`](cosine_similarity.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Stitch Images`](stitch_images.md), [`Polygon Visualization`](polygon_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`Morphological Transformation`](morphological_transformation.md), [`SmolVLM2`](smol_vlm2.md), [`Template Matching`](template_matching.md), [`Image Slicer`](image_slicer.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Barcode Detection`](barcode_detection.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Relative Static Crop`](relative_static_crop.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Clip Comparison`](clip_comparison.md), [`SIFT`](sift.md), [`Dominant Color`](dominant_color.md), [`Moondream2`](moondream2.md), [`Property Definition`](property_definition.md), [`OpenRouter`](open_router.md), [`Identify Changes`](identify_changes.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Overlap Analysis`](overlap_analysis.md), [`Email Notification`](email_notification.md), [`Triangle Visualization`](triangle_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SIFT Comparison`](sift_comparison.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`SIFT Comparison`](sift_comparison.md), [`Grid Visualization`](grid_visualization.md), [`SAM 3`](sam3.md), [`JSON Parser`](json_parser.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Velocity`](velocity.md), [`Image Threshold`](image_threshold.md), [`CogVLM`](cog_vlm.md), [`MQTT Writer`](mqtt_writer.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`OCR Model`](ocr_model.md), [`GLM-OCR`](glmocr.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemma API`](google_gemma_api.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Image Blur`](image_blur.md), [`LMM`](lmm.md), [`SAM 3`](sam3.md), [`OpenAI`](open_ai.md), [`LMM For Classification`](lmm_for_classification.md), [`Time in Zone`](timein_zone.md), [`Path Deviation`](path_deviation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Event Writer`](event_writer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Slack Notification`](slack_notification.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Circle Visualization`](circle_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Camera Focus`](camera_focus.md), [`Byte Tracker`](byte_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`SAM 3`](sam3.md), [`Image Slicer`](image_slicer.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Detections Transformation`](detections_transformation.md), [`Size Measurement`](size_measurement.md), [`Local File Sink`](local_file_sink.md), [`Inner Workflow`](inner_workflow.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Dot Visualization`](dot_visualization.md), [`QR Code Detection`](qr_code_detection.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Depth Estimation`](depth_estimation.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Anthropic Claude`](anthropic_claude.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Qwen3-VL`](qwen3_vl.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Icon Visualization`](icon_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Camera Focus`](camera_focus.md), [`Qwen-VL`](qwen_vl.md), [`Mask Visualization`](mask_visualization.md), [`Text Display`](text_display.md), [`Gaze Detection`](gaze_detection.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Vision OCR`](google_vision_ocr.md), [`OpenAI`](open_ai.md), [`VLM As Detector`](vlm_as_detector.md), [`Identify Outliers`](identify_outliers.md), [`Detection Offset`](detection_offset.md), [`Switch Case`](switch_case.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Overlap Filter`](overlap_filter.md), [`S3 Sink`](s3_sink.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Detection Event Log`](detection_event_log.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Contours`](image_contours.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Camera Calibration`](camera_calibration.md), [`Morphological Transformation`](morphological_transformation.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Path Deviation`](path_deviation.md), [`Background Color Visualization`](background_color_visualization.md), [`Corner Visualization`](corner_visualization.md), [`GeoTag Detection`](geo_tag_detection.md), [`Track Class Lock`](track_class_lock.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Bounding Rectangle`](bounding_rectangle.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Image Preprocessing`](image_preprocessing.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`YOLO-World Model`](yolo_world_model.md), [`Halo Visualization`](halo_visualization.md), [`Label Visualization`](label_visualization.md)
    - outputs: [`Keypoint Visualization`](keypoint_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Blur Visualization`](blur_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Color Visualization`](color_visualization.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Dynamic Zone`](dynamic_zone.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Distance Measurement`](distance_measurement.md), [`Velocity`](velocity.md), [`Trace Visualization`](trace_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Detections Stitch`](detections_stitch.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Mask Visualization`](mask_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Detections Merge`](detections_merge.md), [`Detections Filter`](detections_filter.md), [`Florence-2 Model`](florence2_model.md), [`Dynamic Crop`](dynamic_crop.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Halo Visualization`](halo_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Byte Tracker`](byte_tracker.md), [`Detection Offset`](detection_offset.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Overlap Filter`](overlap_filter.md), [`Polygon Visualization`](polygon_visualization.md), [`Path Deviation`](path_deviation.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Event Writer`](event_writer.md), [`Time in Zone`](timein_zone.md), [`Byte Tracker`](byte_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Circle Visualization`](circle_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Perspective Correction`](perspective_correction.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Camera Focus`](camera_focus.md), [`Byte Tracker`](byte_tracker.md), [`Detections Consensus`](detections_consensus.md), [`Crop Visualization`](crop_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Line Counter`](line_counter.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Path Deviation`](path_deviation.md), [`Detections Combine`](detections_combine.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Background Color Visualization`](background_color_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`GeoTag Detection`](geo_tag_detection.md), [`Corner Visualization`](corner_visualization.md), [`Track Class Lock`](track_class_lock.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Detections Transformation`](detections_transformation.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Size Measurement`](size_measurement.md), [`Overlap Analysis`](overlap_analysis.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Dot Visualization`](dot_visualization.md), [`Time in Zone`](timein_zone.md), [`Label Visualization`](label_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Detections Filter` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `predictions` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Detection predictions to filter (object detection, instance segmentation, or keypoint detection). Each detection is evaluated against the filtering conditions specified in the operations parameter. Only detections that match the filter criteria are included in the output. Supports batch processing, allowing filtering of multiple detection sets simultaneously..
        - `operations_parameters` (*[`*`](../kinds/wildcard.md)*): Dictionary mapping parameter names (referenced in operations) to actual values from the workflow. These parameters provide runtime values used in filtering operations (e.g., class name lists, confidence thresholds). Keys match parameter names used in the operations definition, and values are selectors referencing workflow inputs, step outputs, or computed values. Example: {'classes': '$inputs.allowed_classes', 'threshold': 0.7} where 'classes' and 'threshold' are referenced in the operations..

    - output
    
        - `predictions` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md)]*): Prediction with detected bounding boxes in form of sv.Detections(...) object if `object_detection_prediction` or Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object if `instance_segmentation_prediction` or Prediction with detected bounding boxes and detected keypoints in form of sv.Detections(...) object if `keypoint_detection_prediction`.



??? tip "Example JSON definition of step `Detections Filter` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/detections_filter@v1",
	    "predictions": "$steps.object_detection_model.predictions",
	    "operations": [
	        {
	            "filter_operation": {
	                "statements": [
	                    {
	                        "comparator": {
	                            "type": "in (Sequence)"
	                        },
	                        "left_operand": {
	                            "operations": [
	                                {
	                                    "property_name": "class_name",
	                                    "type": "ExtractDetectionProperty"
	                                }
	                            ],
	                            "type": "DynamicOperand"
	                        },
	                        "right_operand": {
	                            "operand_name": "classes",
	                            "type": "DynamicOperand"
	                        },
	                        "type": "BinaryStatement"
	                    }
	                ],
	                "type": "StatementGroup"
	            },
	            "type": "DetectionsFilter"
	        }
	    ],
	    "operations_parameters": {
	        "classes": "$inputs.classes"
	    }
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    


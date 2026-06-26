
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

    - inputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Qwen-VL`](qwen_vl.md), [`Slack Notification`](slack_notification.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Mask Edge Snap`](mask_edge_snap.md), [`VLM As Detector`](vlm_as_detector.md), [`Template Matching`](template_matching.md), [`Track Class Lock`](track_class_lock.md), [`Dynamic Zone`](dynamic_zone.md), [`Webhook Sink`](webhook_sink.md), [`SAM 3`](sam3.md), [`Contrast Enhancement`](contrast_enhancement.md), [`QR Code Generator`](qr_code_generator.md), [`Bounding Rectangle`](bounding_rectangle.md), [`SIFT`](sift.md), [`Path Deviation`](path_deviation.md), [`Florence-2 Model`](florence2_model.md), [`Delta Filter`](delta_filter.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Text Display`](text_display.md), [`MQTT Writer`](mqtt_writer.md), [`Gaze Detection`](gaze_detection.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Expression`](expression.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Time in Zone`](timein_zone.md), [`Time in Zone`](timein_zone.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Image Slicer`](image_slicer.md), [`Background Subtraction`](background_subtraction.md), [`Image Preprocessing`](image_preprocessing.md), [`Byte Tracker`](byte_tracker.md), [`Email Notification`](email_notification.md), [`Line Counter`](line_counter.md), [`Reference Path Visualization`](reference_path_visualization.md), [`LMM`](lmm.md), [`Distance Measurement`](distance_measurement.md), [`Detections Consensus`](detections_consensus.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`SAM 3`](sam3.md), [`Time in Zone`](timein_zone.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Event Writer`](event_writer.md), [`Image Blur`](image_blur.md), [`Google Vision OCR`](google_vision_ocr.md), [`Byte Tracker`](byte_tracker.md), [`Qwen3-VL`](qwen3_vl.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`CSV Formatter`](csv_formatter.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Detections Merge`](detections_merge.md), [`OpenAI`](open_ai.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Detections Transformation`](detections_transformation.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`LMM For Classification`](lmm_for_classification.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Property Definition`](property_definition.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Object Detection Model`](object_detection_model.md), [`Seg Preview`](seg_preview.md), [`Cache Get`](cache_get.md), [`Camera Calibration`](camera_calibration.md), [`Dominant Color`](dominant_color.md), [`Detections Filter`](detections_filter.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`CogVLM`](cog_vlm.md), [`SORT Tracker`](sort_tracker.md), [`OpenAI`](open_ai.md), [`Identify Outliers`](identify_outliers.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Depth Estimation`](depth_estimation.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Stack`](image_stack.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Google Gemma`](google_gemma.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Detections Stitch`](detections_stitch.md), [`Buffer`](buffer.md), [`Crop Visualization`](crop_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Camera Focus`](camera_focus.md), [`JSON Parser`](json_parser.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`GLM-OCR`](glmocr.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Dimension Collapse`](dimension_collapse.md), [`Continue If`](continue_if.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`VLM As Detector`](vlm_as_detector.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Threshold`](image_threshold.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen3.5`](qwen3.5.md), [`OpenAI`](open_ai.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Detections Combine`](detections_combine.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Cosine Similarity`](cosine_similarity.md), [`Florence-2 Model`](florence2_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Halo Visualization`](halo_visualization.md), [`SAM 3`](sam3.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Inner Workflow`](inner_workflow.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Byte Tracker`](byte_tracker.md), [`Clip Comparison`](clip_comparison.md), [`Halo Visualization`](halo_visualization.md), [`Label Visualization`](label_visualization.md), [`Detection Offset`](detection_offset.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Morphological Transformation`](morphological_transformation.md), [`PLC Writer`](plc_writer.md), [`Google Gemini`](google_gemini.md), [`PLC Reader`](plc_reader.md), [`Polygon Visualization`](polygon_visualization.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Moondream2`](moondream2.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Google Gemini`](google_gemini.md), [`EasyOCR`](easy_ocr.md), [`Overlap Analysis`](overlap_analysis.md), [`Motion Detection`](motion_detection.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Switch Case`](switch_case.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Data Aggregator`](data_aggregator.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Local File Sink`](local_file_sink.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Rate Limiter`](rate_limiter.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Slicer`](image_slicer.md), [`Identify Changes`](identify_changes.md), [`S3 Sink`](s3_sink.md), [`Cache Set`](cache_set.md), [`Size Measurement`](size_measurement.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Path Deviation`](path_deviation.md), [`Contrast Equalization`](contrast_equalization.md), [`Morphological Transformation`](morphological_transformation.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Barcode Detection`](barcode_detection.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SmolVLM2`](smol_vlm2.md), [`Email Notification`](email_notification.md), [`OpenRouter`](open_router.md), [`Grid Visualization`](grid_visualization.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Velocity`](velocity.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Clip Comparison`](clip_comparison.md), [`Overlap Filter`](overlap_filter.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Corner Visualization`](corner_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Line Counter`](line_counter.md), [`Current Time`](current_time.md), [`Stitch Images`](stitch_images.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Icon Visualization`](icon_visualization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`OCR Model`](ocr_model.md), [`Perspective Correction`](perspective_correction.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Pixel Color Count`](pixel_color_count.md), [`QR Code Detection`](qr_code_detection.md), [`Detection Event Log`](detection_event_log.md), [`Image Contours`](image_contours.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md)
    - outputs: [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Triangle Visualization`](triangle_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Detections Combine`](detections_combine.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Detections Merge`](detections_merge.md), [`Florence-2 Model`](florence2_model.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Detections Transformation`](detections_transformation.md), [`Camera Focus`](camera_focus.md), [`Track Class Lock`](track_class_lock.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Halo Visualization`](halo_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Dynamic Crop`](dynamic_crop.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Velocity`](velocity.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Label Visualization`](label_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Detection Offset`](detection_offset.md), [`Florence-2 Model`](florence2_model.md), [`Path Deviation`](path_deviation.md), [`Overlap Filter`](overlap_filter.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Detections Filter`](detections_filter.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Polygon Visualization`](polygon_visualization.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Background Color Visualization`](background_color_visualization.md), [`Overlap Analysis`](overlap_analysis.md), [`Corner Visualization`](corner_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Time in Zone`](timein_zone.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Time in Zone`](timein_zone.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Polygon Visualization`](polygon_visualization.md), [`Line Counter`](line_counter.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Keypoint Visualization`](keypoint_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Byte Tracker`](byte_tracker.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Perspective Correction`](perspective_correction.md), [`Crop Visualization`](crop_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Line Counter`](line_counter.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Distance Measurement`](distance_measurement.md), [`Detections Consensus`](detections_consensus.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Size Measurement`](size_measurement.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Time in Zone`](timein_zone.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Event Writer`](event_writer.md), [`Path Deviation`](path_deviation.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Detections Filter` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `predictions` (*Union[[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md)]*): Detection predictions to filter (object detection, instance segmentation, or keypoint detection). Each detection is evaluated against the filtering conditions specified in the operations parameter. Only detections that match the filter criteria are included in the output. Supports batch processing, allowing filtering of multiple detection sets simultaneously..
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



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

    - inputs: [`Absolute Static Crop`](absolute_static_crop.md), [`Gaze Detection`](gaze_detection.md), [`Byte Tracker`](byte_tracker.md), [`Triangle Visualization`](triangle_visualization.md), [`EasyOCR`](easy_ocr.md), [`Detection Event Log`](detection_event_log.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Track Class Lock`](track_class_lock.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Continue If`](continue_if.md), [`Dominant Color`](dominant_color.md), [`GLM-OCR`](glmocr.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`VLM As Detector`](vlm_as_detector.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Distance Measurement`](distance_measurement.md), [`SmolVLM2`](smol_vlm2.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Corner Visualization`](corner_visualization.md), [`Overlap Analysis`](overlap_analysis.md), [`Buffer`](buffer.md), [`Template Matching`](template_matching.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Velocity`](velocity.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Morphological Transformation`](morphological_transformation.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Calibration`](camera_calibration.md), [`Clip Comparison`](clip_comparison.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`OpenAI`](open_ai.md), [`Halo Visualization`](halo_visualization.md), [`Detections Combine`](detections_combine.md), [`Contrast Equalization`](contrast_equalization.md), [`CSV Formatter`](csv_formatter.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Cache Set`](cache_set.md), [`Delta Filter`](delta_filter.md), [`Blur Visualization`](blur_visualization.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Background Subtraction`](background_subtraction.md), [`Object Detection Model`](object_detection_model.md), [`Path Deviation`](path_deviation.md), [`SAM 3`](sam3.md), [`QR Code Detection`](qr_code_detection.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Stack`](image_stack.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Qwen3-VL`](qwen3_vl.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Camera Focus`](camera_focus.md), [`Florence-2 Model`](florence2_model.md), [`Time in Zone`](timein_zone.md), [`Dynamic Crop`](dynamic_crop.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Perspective Correction`](perspective_correction.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`CogVLM`](cog_vlm.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Byte Tracker`](byte_tracker.md), [`LMM For Classification`](lmm_for_classification.md), [`Florence-2 Model`](florence2_model.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Image Blur`](image_blur.md), [`Label Visualization`](label_visualization.md), [`Seg Preview`](seg_preview.md), [`Path Deviation`](path_deviation.md), [`Object Detection Model`](object_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Cosine Similarity`](cosine_similarity.md), [`Pixel Color Count`](pixel_color_count.md), [`Trace Visualization`](trace_visualization.md), [`Email Notification`](email_notification.md), [`SIFT Comparison`](sift_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Current Time`](current_time.md), [`Property Definition`](property_definition.md), [`Image Contours`](image_contours.md), [`Camera Focus`](camera_focus.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Moondream2`](moondream2.md), [`Data Aggregator`](data_aggregator.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemma`](google_gemma.md), [`Detections Merge`](detections_merge.md), [`Google Gemini`](google_gemini.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Email Notification`](email_notification.md), [`Byte Tracker`](byte_tracker.md), [`SAM 3`](sam3.md), [`Detections Stitch`](detections_stitch.md), [`Text Display`](text_display.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Qwen3.5`](qwen3.5.md), [`OpenAI`](open_ai.md), [`Time in Zone`](timein_zone.md), [`SIFT Comparison`](sift_comparison.md), [`SIFT`](sift.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Mask Visualization`](mask_visualization.md), [`Detections Filter`](detections_filter.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Qwen-VL`](qwen_vl.md), [`Cache Get`](cache_get.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`YOLO-World Model`](yolo_world_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Stitch Images`](stitch_images.md), [`Google Gemini`](google_gemini.md), [`QR Code Generator`](qr_code_generator.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Line Counter`](line_counter.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Rate Limiter`](rate_limiter.md), [`Switch Case`](switch_case.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`JSON Parser`](json_parser.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`LMM`](lmm.md), [`Inner Workflow`](inner_workflow.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Dimension Collapse`](dimension_collapse.md), [`Motion Detection`](motion_detection.md), [`Detection Offset`](detection_offset.md), [`Google Vision OCR`](google_vision_ocr.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Identify Changes`](identify_changes.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Expression`](expression.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`MQTT Writer`](mqtt_writer.md), [`PLC Reader`](plc_reader.md), [`Image Threshold`](image_threshold.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Google Gemini`](google_gemini.md), [`Background Color Visualization`](background_color_visualization.md), [`Identify Outliers`](identify_outliers.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Dynamic Zone`](dynamic_zone.md), [`Image Slicer`](image_slicer.md), [`Overlap Filter`](overlap_filter.md), [`Polygon Visualization`](polygon_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Event Writer`](event_writer.md), [`Line Counter`](line_counter.md), [`Polygon Visualization`](polygon_visualization.md), [`Clip Comparison`](clip_comparison.md), [`OCR Model`](ocr_model.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Circle Visualization`](circle_visualization.md), [`Slack Notification`](slack_notification.md), [`Contrast Enhancement`](contrast_enhancement.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Grid Visualization`](grid_visualization.md), [`Time in Zone`](timein_zone.md), [`Barcode Detection`](barcode_detection.md), [`Local File Sink`](local_file_sink.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`OpenAI`](open_ai.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`SORT Tracker`](sort_tracker.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Preprocessing`](image_preprocessing.md), [`Detections Consensus`](detections_consensus.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dot Visualization`](dot_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Halo Visualization`](halo_visualization.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`OpenRouter`](open_router.md), [`S3 Sink`](s3_sink.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Size Measurement`](size_measurement.md), [`Depth Estimation`](depth_estimation.md), [`Icon Visualization`](icon_visualization.md), [`GeoTag Detection`](geo_tag_detection.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`PLC Writer`](plc_writer.md), [`VLM As Classifier`](vlm_as_classifier.md)
    - outputs: [`Path Deviation`](path_deviation.md), [`Detections Filter`](detections_filter.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Triangle Visualization`](triangle_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Crop Visualization`](crop_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Overlap Filter`](overlap_filter.md), [`Track Class Lock`](track_class_lock.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Time in Zone`](timein_zone.md), [`Polygon Visualization`](polygon_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Dynamic Crop`](dynamic_crop.md), [`Event Writer`](event_writer.md), [`Perspective Correction`](perspective_correction.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Line Counter`](line_counter.md), [`Polygon Visualization`](polygon_visualization.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Circle Visualization`](circle_visualization.md), [`Line Counter`](line_counter.md), [`Distance Measurement`](distance_measurement.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Time in Zone`](timein_zone.md), [`Byte Tracker`](byte_tracker.md), [`Florence-2 Model`](florence2_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Overlap Analysis`](overlap_analysis.md), [`SORT Tracker`](sort_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Corner Visualization`](corner_visualization.md), [`Label Visualization`](label_visualization.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Path Deviation`](path_deviation.md), [`Velocity`](velocity.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Trace Visualization`](trace_visualization.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Detections Consensus`](detections_consensus.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Dot Visualization`](dot_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Detection Offset`](detection_offset.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Camera Focus`](camera_focus.md), [`Halo Visualization`](halo_visualization.md), [`Detections Combine`](detections_combine.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Detections Merge`](detections_merge.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Byte Tracker`](byte_tracker.md), [`Detections Stitch`](detections_stitch.md), [`Size Measurement`](size_measurement.md), [`Icon Visualization`](icon_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Time in Zone`](timein_zone.md), [`GeoTag Detection`](geo_tag_detection.md), [`Background Color Visualization`](background_color_visualization.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Mask Visualization`](mask_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Detections Filter` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `predictions` (*Union[[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md)]*): Detection predictions to filter (object detection, instance segmentation, or keypoint detection). Each detection is evaluated against the filtering conditions specified in the operations parameter. Only detections that match the filter criteria are included in the output. Supports batch processing, allowing filtering of multiple detection sets simultaneously..
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


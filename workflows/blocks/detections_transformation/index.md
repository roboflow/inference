
# Detections Transformation



??? "Class: `DetectionsTransformationBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/transformations/detections_transformation/v1.py">inference.core.workflows.core_steps.transformations.detections_transformation.v1.DetectionsTransformationBlockV1</a>
    



Apply customizable transformations to detection predictions using UQL (Query Language) operation chains, enabling flexible modification of bounding boxes, filtering detections, extracting properties, resizing boxes, and other detection manipulations through configurable operation sequences for advanced detection processing workflows.

## How This Block Works

This block transforms detection predictions by applying a chain of UQL operations that can modify, filter, extract, or manipulate detection data. The block:

1. Receives detection predictions (object detection, instance segmentation, or keypoint detection) and a list of UQL operations to apply
2. Validates that operations_parameters doesn't contain reserved parameter names
3. Builds an operations chain from the provided UQL operation definitions, creating a sequence of transformations to apply in order
4. Separates operations_parameters into batch parameters (aligned with predictions) and non-batch parameters (applied to all predictions)
5. Processes each prediction batch by applying the operations chain:
   - Zips predictions with batch parameters to align data per batch item
   - Combines batch and non-batch parameters into evaluation parameters for each prediction
   - Applies the operations chain to the detections with the combined parameters
   - Validates that the output is still sv.Detections (operations must preserve detection type)
6. Returns the transformed detections for each input batch

The block supports a wide variety of UQL operations including filtering (DetectionsFilter), property extraction (ExtractDetectionProperty), bounding box transformations (resizing, scaling), and other detection manipulations. Operations are applied sequentially, allowing complex transformations through operation chaining. The block validates that transformations preserve the detection type, ensuring outputs remain compatible with other detection-processing blocks. Batch and non-batch parameters enable flexible operation parameterization, supporting both per-detection and global parameter values.

## Common Use Cases

- **Advanced Detection Filtering**: Apply complex filtering logic to detection predictions (e.g., filter detections by class names using conditional statements, filter by confidence thresholds with multiple conditions, apply custom filtering criteria based on detection properties), enabling sophisticated detection selection workflows
- **Bounding Box Transformations**: Modify bounding box sizes, positions, or properties (e.g., resize bounding boxes proportionally, scale boxes by percentage, adjust box coordinates, transform box dimensions), enabling flexible bounding box manipulation
- **Property Extraction and Filtering**: Extract detection properties and filter based on extracted values (e.g., extract class names and filter by class lists, extract confidence scores and filter by thresholds, extract properties for conditional processing), enabling property-based detection processing
- **Multi-Conditional Processing**: Apply complex conditional transformations based on multiple detection criteria (e.g., transform detections based on class and confidence combinations, apply different operations for different detection types, conditionally modify detections based on multiple properties), enabling sophisticated conditional detection processing
- **Detection Data Enrichment**: Extract and add properties to detections for downstream processing (e.g., extract class names for filtering, compute detection properties, add metadata to detections), enabling enriched detection data for complex workflows
- **Custom Detection Manipulation**: Apply custom transformations not available in dedicated blocks (e.g., complex multi-step detection modifications, custom filtering and transformation combinations, specialized detection processing workflows), enabling flexible custom detection processing

## Connecting to Other Blocks

This block receives detection predictions and produces transformed detections:

- **After detection blocks** (e.g., Object Detection, Instance Segmentation, Keypoint Detection) to apply custom transformations, filtering, or modifications to detection predictions, enabling flexible detection processing workflows
- **Before dynamic crop blocks** to filter or modify detections before cropping (e.g., filter detections by class before cropping, transform box sizes before cropping, extract specific detections for cropping), enabling optimized region extraction workflows
- **Before classification or analysis blocks** to prepare detections with custom filtering or transformations (e.g., filter detections for specific analysis, transform boxes for compatibility, prepare detections with custom criteria), enabling customized detection preparation
- **In multi-stage detection workflows** where detections need custom transformations between stages (e.g., filter and transform initial detections before secondary processing, apply custom modifications between detection stages, conditionally process detections based on criteria), enabling sophisticated multi-stage workflows
- **Before visualization blocks** to filter or transform detections for display (e.g., filter detections for visualization, transform boxes for presentation, customize detections for display purposes), enabling optimized visual outputs
- **After detection blocks and before other transformation blocks** to apply custom logic between transformations (e.g., filter after detection and before cropping, transform between detection stages, apply conditional modifications), enabling complex transformation pipelines with custom logic


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/detections_transformation@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `operations` | `List[Union[ClassificationPropertyExtract, ConvertDictionaryToJSON, ConvertImageToBase64, ConvertImageToJPEG, DetectionsFilter, DetectionsOffset, DetectionsPropertyExtract, DetectionsRename, DetectionsSelection, DetectionsShift, DetectionsToDictionary, Divide, ExtractDetectionProperty, ExtractFrameMetadata, ExtractImageProperty, LookupTable, Multiply, NumberRound, NumericSequenceAggregate, PickDetectionsByParentClass, RandomNumber, SequenceAggregate, SequenceApply, SequenceElementsCount, SequenceLength, SequenceMap, SortDetections, StringMatches, StringSubSequence, StringToLowerCase, StringToUpperCase, TimestampToISOFormat, ToBoolean, ToNumber, ToString]]` | List of UQL (Query Language) operations to apply sequentially to the detections. Operations are executed in order, with each operation receiving the output of the previous operation. Supported operations include DetectionsFilter (filtering detections by conditions), ExtractDetectionProperty (extracting properties from detections), bounding box transformations (resizing, scaling), and other UQL operations that accept and return sv.Detections. Operations can be parameterized using operations_parameters. The operations chain must transform sv.Detections to sv.Detections (type must be preserved).. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Detections Transformation` in version `v1`.

    - inputs: [`Anthropic Claude`](anthropic_claude.md), [`Image Slicer`](image_slicer.md), [`Dynamic Crop`](dynamic_crop.md), [`Mask Area Measurement`](mask_area_measurement.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Cache Get`](cache_get.md), [`SIFT Comparison`](sift_comparison.md), [`Stitch Images`](stitch_images.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Distance Measurement`](distance_measurement.md), [`EasyOCR`](easy_ocr.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Track Class Lock`](track_class_lock.md), [`Florence-2 Model`](florence2_model.md), [`Gaze Detection`](gaze_detection.md), [`YOLO-World Model`](yolo_world_model.md), [`Dominant Color`](dominant_color.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Byte Tracker`](byte_tracker.md), [`PLC Reader`](plc_reader.md), [`Clip Comparison`](clip_comparison.md), [`CogVLM`](cog_vlm.md), [`PP-OCR`](ppocr.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Morphological Transformation`](morphological_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OpenAI`](open_ai.md), [`Time in Zone`](timein_zone.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Image Contours`](image_contours.md), [`SAM 3`](sam3.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`QR Code Generator`](qr_code_generator.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Path Deviation`](path_deviation.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`GeoTag Detection`](geo_tag_detection.md), [`Camera Focus`](camera_focus.md), [`Inner Workflow`](inner_workflow.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`VLM As Detector`](vlm_as_detector.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`MQTT Writer`](mqtt_writer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Local File Sink`](local_file_sink.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Google Gemini`](google_gemini.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Event Writer`](event_writer.md), [`Depth Estimation`](depth_estimation.md), [`Seg Preview`](seg_preview.md), [`Byte Tracker`](byte_tracker.md), [`Google Gemini`](google_gemini.md), [`OpenAI`](open_ai.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Pixel Color Count`](pixel_color_count.md), [`Object Detection Model`](object_detection_model.md), [`Webhook Sink`](webhook_sink.md), [`Halo Visualization`](halo_visualization.md), [`Buffer`](buffer.md), [`Switch Case`](switch_case.md), [`Dot Visualization`](dot_visualization.md), [`Image Stack`](image_stack.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Image Slicer`](image_slicer.md), [`Time in Zone`](timein_zone.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Morphological Transformation`](morphological_transformation.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Halo Visualization`](halo_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemma`](google_gemma.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Qwen3.5`](qwen3.5.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Motion Detection`](motion_detection.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Detections Consensus`](detections_consensus.md), [`Overlap Analysis`](overlap_analysis.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Image Blur`](image_blur.md), [`Grid Visualization`](grid_visualization.md), [`Blur Visualization`](blur_visualization.md), [`GLM-OCR`](glmocr.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Vision OCR`](google_vision_ocr.md), [`Polygon Visualization`](polygon_visualization.md), [`Data Aggregator`](data_aggregator.md), [`SIFT`](sift.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Text Display`](text_display.md), [`JSON Parser`](json_parser.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Identify Changes`](identify_changes.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`S3 Sink`](s3_sink.md), [`Line Counter`](line_counter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SAM 3`](sam3.md), [`Image Preprocessing`](image_preprocessing.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Time in Zone`](timein_zone.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Mask Edge Snap`](mask_edge_snap.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Path Deviation`](path_deviation.md), [`QR Code Detection`](qr_code_detection.md), [`Expression`](expression.md), [`Absolute Static Crop`](absolute_static_crop.md), [`OpenAI`](open_ai.md), [`Email Notification`](email_notification.md), [`Frame Delay`](frame_delay.md), [`Cosine Similarity`](cosine_similarity.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Dynamic Zone`](dynamic_zone.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`Byte Tracker`](byte_tracker.md), [`Detections Transformation`](detections_transformation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Property Definition`](property_definition.md), [`Continue If`](continue_if.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Slack Notification`](slack_notification.md), [`Camera Calibration`](camera_calibration.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`VLM As Detector`](vlm_as_detector.md), [`SORT Tracker`](sort_tracker.md), [`Camera Focus`](camera_focus.md), [`Corner Visualization`](corner_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Rate Limiter`](rate_limiter.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Color Visualization`](color_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Object Detection Model`](object_detection_model.md), [`SAM 3`](sam3.md), [`Image Threshold`](image_threshold.md), [`Barcode Detection`](barcode_detection.md), [`Detections Merge`](detections_merge.md), [`Current Time`](current_time.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Florence-2 Model`](florence2_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`SmolVLM2`](smol_vlm2.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Moondream2`](moondream2.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Label Visualization`](label_visualization.md), [`Detection Offset`](detection_offset.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Line Counter`](line_counter.md), [`Identify Outliers`](identify_outliers.md), [`Cache Set`](cache_set.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Trace Visualization`](trace_visualization.md), [`SAM 3 Interactive`](sam3_interactive.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Detections Filter`](detections_filter.md), [`LMM For Classification`](lmm_for_classification.md), [`Velocity`](velocity.md), [`Mask Visualization`](mask_visualization.md), [`Template Matching`](template_matching.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Environment Secrets Store`](environment_secrets_store.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Google Gemini`](google_gemini.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`LMM`](lmm.md), [`Detections Combine`](detections_combine.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`OCR Model`](ocr_model.md), [`Circle Visualization`](circle_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Email Notification`](email_notification.md), [`Cosmos 3`](cosmos3.md), [`Polygon Visualization`](polygon_visualization.md), [`Qwen-VL`](qwen_vl.md), [`PLC Writer`](plc_writer.md), [`Crop Visualization`](crop_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Qwen3-VL`](qwen3_vl.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Size Measurement`](size_measurement.md), [`Icon Visualization`](icon_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Delta Filter`](delta_filter.md), [`CSV Formatter`](csv_formatter.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Detections Stitch`](detections_stitch.md), [`Background Color Visualization`](background_color_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Background Subtraction`](background_subtraction.md), [`Contrast Equalization`](contrast_equalization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenRouter`](open_router.md), [`OpenAI`](open_ai.md), [`SIFT Comparison`](sift_comparison.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Overlap Filter`](overlap_filter.md)
    - outputs: [`Trace Visualization`](trace_visualization.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Time in Zone`](timein_zone.md), [`Dynamic Crop`](dynamic_crop.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Detections Filter`](detections_filter.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Velocity`](velocity.md), [`Halo Visualization`](halo_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Path Deviation`](path_deviation.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Frame Delay`](frame_delay.md), [`Distance Measurement`](distance_measurement.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Track Class Lock`](track_class_lock.md), [`Florence-2 Model`](florence2_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Dynamic Zone`](dynamic_zone.md), [`Detections Combine`](detections_combine.md), [`Detections Transformation`](detections_transformation.md), [`Byte Tracker`](byte_tracker.md), [`Time in Zone`](timein_zone.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Circle Visualization`](circle_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Byte Tracker`](byte_tracker.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Halo Visualization`](halo_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Camera Focus`](camera_focus.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Size Measurement`](size_measurement.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Color Visualization`](color_visualization.md), [`Detections Consensus`](detections_consensus.md), [`Overlap Analysis`](overlap_analysis.md), [`Triangle Visualization`](triangle_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Detections Stitch`](detections_stitch.md), [`Background Color Visualization`](background_color_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Time in Zone`](timein_zone.md), [`Blur Visualization`](blur_visualization.md), [`Detections Merge`](detections_merge.md), [`Florence-2 Model`](florence2_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Path Deviation`](path_deviation.md), [`GeoTag Detection`](geo_tag_detection.md), [`Perspective Correction`](perspective_correction.md), [`Polygon Visualization`](polygon_visualization.md), [`Label Visualization`](label_visualization.md), [`Detection Offset`](detection_offset.md), [`Line Counter`](line_counter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Overlap Filter`](overlap_filter.md), [`Event Writer`](event_writer.md), [`Line Counter`](line_counter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Detections Transformation` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `predictions` (*Union[[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md)]*): Detection predictions to transform using UQL operations. Supports object detection, instance segmentation, or keypoint detection predictions. The detections will be transformed by the operations chain defined in the operations field. All transformations must preserve the detection type (output must remain sv.Detections). The block processes batch inputs and applies transformations per batch item..
        - `operations_parameters` (*[`*`](../kinds/wildcard.md)*): Dictionary mapping parameter names (used in operations) to workflow data sources or values. Parameters are referenced in operations (e.g., in conditional statements, filter operations) and provided at runtime. Supports both batch parameters (aligned with predictions, one value per batch item) and non-batch parameters (same value for all batch items). Parameters are automatically separated into batch and non-batch based on their data structure. Cannot use reserved parameter names. Use this to parameterize operations dynamically (e.g., provide class lists for filtering, provide thresholds for conditions, supply values for operations that need runtime parameters)..

    - output
    
        - `predictions` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md)]*): Prediction with detected bounding boxes in form of sv.Detections(...) object if `object_detection_prediction` or Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object if `instance_segmentation_prediction` or Prediction with detected bounding boxes and detected keypoints in form of sv.Detections(...) object if `keypoint_detection_prediction`.



??? tip "Example JSON definition of step `Detections Transformation` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/detections_transformation@v1",
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


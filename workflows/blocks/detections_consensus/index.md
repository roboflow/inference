
# Detections Consensus



??? "Class: `DetectionsConsensusBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/fusion/detections_consensus/v1.py">inference.core.workflows.core_steps.fusion.detections_consensus.v1.DetectionsConsensusBlockV1</a>
    



Combine detection predictions from multiple models using a majority vote consensus strategy, merging overlapping detections that receive sufficient votes from different models and aggregating their properties (confidence scores, bounding boxes, masks) into unified consensus detections with improved accuracy and reliability.

## How This Block Works

This block fuses predictions from multiple detection models by requiring agreement (consensus) among models before accepting detections. The block:

1. Takes detection predictions from multiple model sources (object detection, instance segmentation, or keypoint detection) as input
2. Matches detections from different models that overlap spatially by calculating Intersection over Union (IoU) between bounding boxes
3. Compares overlapping detections against an IoU threshold to determine if they represent the same object
4. Counts "votes" for each detection by finding matching detections from other models (subject to class-awareness if enabled)
5. Requires a minimum number of votes (`required_votes`) before accepting a detection as part of the consensus output
6. Aggregates properties of matching detections using configurable modes:
   - **Confidence aggregation**: Combines confidence scores using average, max, or min
   - **Coordinates aggregation**: Merges bounding boxes using average (mean coordinates), max (largest box), or min (smallest box)
   - **Mask aggregation** (for instance segmentation): Combines masks using union, intersection, max, or min
   - **Class selection**: Chooses class name based on majority vote (average), highest confidence (max), or lowest confidence (min)
7. Filters detections based on optional criteria (specific classes to consider, minimum confidence threshold)
8. Determines object presence by checking if the required number of objects (per class or total) are present in consensus results
9. Returns merged consensus detections, object presence indicators, and presence confidence scores

The block enables class-aware or class-agnostic matching: when `class_aware` is true, only detections with matching class names are considered for voting; when false, any overlapping detections (regardless of class) contribute votes. The consensus mechanism helps reduce false positives (detections seen by only one model) and improves reliability by requiring multiple models to agree on object presence. Aggregation modes allow flexibility in how overlapping detections are combined, balancing between conservative (intersection, min) and inclusive (union, max) strategies.

## Common Use Cases

- **Multi-Model Ensemble**: Combine predictions from multiple specialized models (e.g., one optimized for people, another for vehicles) to improve overall detection accuracy, leveraging strengths of different models while filtering out detections that only one model sees
- **Reducing False Positives**: Require consensus from multiple models before accepting detections (e.g., require 2 out of 3 models to detect an object), reducing false positives by filtering out detections seen by only one model
- **Improving Detection Reliability**: Use majority voting to increase confidence in detections (e.g., merge overlapping detections from 3 models, keeping only those with 2+ votes), ensuring only high-confidence, multi-model-agreed detections are retained
- **Object Presence Detection**: Determine if specific objects are present based on consensus (e.g., check if at least 2 "person" detections exist across models, use aggregated confidence to determine presence), enabling robust object presence checking with configurable thresholds
- **Class-Specific Consensus**: Apply different consensus requirements per class (e.g., require 3 votes for "car" but only 2 for "person"), allowing stricter criteria for critical objects while being more lenient for common detections
- **Specialized Model Fusion**: Combine general-purpose and specialized models (e.g., general object detector + specialized license plate detector), creating a unified detection system that benefits from both broad coverage and specific expertise

## Connecting to Other Blocks

The consensus predictions from this block can be connected to:

- **Multiple detection model blocks** (e.g., Object Detection Model, Instance Segmentation Model) to receive predictions from different models that are fused into consensus detections based on majority voting and spatial overlap matching
- **Visualization blocks** (e.g., Bounding Box Visualization, Polygon Visualization, Label Visualization) to display the merged consensus detections, showing unified results from multiple models with improved accuracy
- **Counting and analytics blocks** (e.g., Line Counter, Time in Zone, Velocity) to count or analyze consensus detections, providing more reliable metrics based on multi-model agreement
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload, Webhook Sink) to save or transmit consensus detection results, storing fused predictions that represent multi-model agreement
- **Flow control blocks** (e.g., Continue If) to conditionally trigger downstream processing based on `object_present` indicators or `presence_confidence` scores, enabling workflows that respond to multi-model consensus on object presence
- **Filtering blocks** (e.g., Detections Filter) to further refine consensus detections based on additional criteria, enabling multi-stage filtering after consensus fusion


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/detections_consensus@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `required_votes` | `int` | Minimum number of votes (matching detections from different models) required to accept a detection in the consensus output. Detections that receive fewer votes than this threshold are filtered out. For example, if set to 2, at least 2 models must detect an overlapping object (above IoU threshold) for it to appear in the consensus results. Higher values create stricter consensus requirements, reducing false positives but potentially missing detections seen by fewer models.. | ✅ |
| `class_aware` | `bool` | If true, only detections with matching class names from different models are considered as votes for the same object. If false, any overlapping detections (regardless of class) contribute votes. Class-aware mode is more conservative and ensures class consistency in consensus, while class-agnostic mode allows voting across different classes but may merge detections of different object types.. | ✅ |
| `iou_threshold` | `float` | Intersection over Union (IoU) threshold for considering detections from different models as matching the same object. Detections with IoU above this threshold are considered overlapping and contribute votes to each other. Lower values (e.g., 0.2) are more lenient and match detections with less overlap, while higher values (e.g., 0.5) require stronger spatial overlap for matching. Typical values range from 0.2 to 0.5.. | ✅ |
| `confidence` | `float` | Confidence threshold applied to merged consensus detections. Only detections with aggregated confidence scores above this threshold are included in the output. Set to 0.0 to disable confidence filtering. Higher values filter out low-confidence consensus detections, improving output quality at the cost of potentially removing valid but lower-confidence detections.. | ✅ |
| `classes_to_consider` | `List[str]` | Optional list of class names to include in the consensus procedure. If provided, only detections of these classes are considered for voting and merging; all other classes are filtered out before consensus matching. Use this to focus consensus on specific object types while ignoring irrelevant detections. If None, all classes participate in consensus.. | ✅ |
| `required_objects` | `Optional[Dict[str, int], int]` | Optional minimum number of objects required to determine object presence. Can be an integer (total objects across all classes) or a dictionary mapping class names to per-class minimum counts. Used in conjunction with object_present output to determine if sufficient objects of each class are detected. For example, 3 means at least 3 total objects must be present, while {'person': 2, 'car': 1} requires at least 2 persons and 1 car. If None, object presence is determined solely by whether any consensus detections exist.. | ✅ |
| `presence_confidence_aggregation` | `AggregationMode` | Aggregation mode for calculating presence confidence scores. Determines how confidence values are combined when computing object presence confidence: 'average' (mean confidence), 'max' (highest confidence), or 'min' (lowest confidence). This mode applies to the presence_confidence output which indicates confidence that required objects are present.. | ❌ |
| `detections_merge_confidence_aggregation` | `AggregationMode` | Aggregation mode for merging confidence scores of overlapping detections. 'average' computes mean confidence (majority vote approach), 'max' uses the highest confidence among matching detections, 'min' uses the lowest confidence. For class selection, 'average' represents majority vote (most common class), 'max' selects class from detection with highest confidence, 'min' selects class from detection with lowest confidence.. | ❌ |
| `detections_merge_coordinates_aggregation` | `AggregationMode` | Aggregation mode for merging bounding box coordinates of overlapping detections. 'average' computes mean coordinates from all matching boxes (balanced approach), 'max' takes the largest box (most inclusive), 'min' takes the smallest box (most conservative). This mode only applies to bounding boxes; mask aggregation uses detections_merge_mask_aggregation instead.. | ❌ |
| `detections_merge_mask_aggregation` | `MaskAggregationMode` | Aggregation mode for merging segmentation masks of overlapping detections. 'union' combines all masks into the largest possible area (most inclusive), 'intersection' takes only the overlapping region (most conservative), 'max' selects the largest mask, 'min' selects the smallest mask. This mode applies only to instance segmentation detections with masks; bounding box detections use detections_merge_coordinates_aggregation instead.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Detections Consensus` in version `v1`.

    - inputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Google Vision OCR`](google_vision_ocr.md), [`Byte Tracker`](byte_tracker.md), [`VLM As Detector`](vlm_as_detector.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Anthropic Claude`](anthropic_claude.md), [`Qwen3.5`](qwen3.5.md), [`OpenAI`](open_ai.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Qwen3-VL`](qwen3_vl.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Qwen-VL`](qwen_vl.md), [`Detections Combine`](detections_combine.md), [`Slack Notification`](slack_notification.md), [`Email Notification`](email_notification.md), [`SmolVLM2`](smol_vlm2.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`OpenRouter`](open_router.md), [`Mask Edge Snap`](mask_edge_snap.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Detections Merge`](detections_merge.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`VLM As Detector`](vlm_as_detector.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections Transformation`](detections_transformation.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`Template Matching`](template_matching.md), [`Track Class Lock`](track_class_lock.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Dynamic Zone`](dynamic_zone.md), [`Webhook Sink`](webhook_sink.md), [`Anthropic Claude`](anthropic_claude.md), [`SAM 3`](sam3.md), [`Dynamic Crop`](dynamic_crop.md), [`Anthropic Claude`](anthropic_claude.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Velocity`](velocity.md), [`SAM 3`](sam3.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Byte Tracker`](byte_tracker.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Clip Comparison`](clip_comparison.md), [`Detection Offset`](detection_offset.md), [`Object Detection Model`](object_detection_model.md), [`Path Deviation`](path_deviation.md), [`Florence-2 Model`](florence2_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Seg Preview`](seg_preview.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Clip Comparison`](clip_comparison.md), [`MQTT Writer`](mqtt_writer.md), [`PLC Writer`](plc_writer.md), [`Overlap Filter`](overlap_filter.md), [`Gaze Detection`](gaze_detection.md), [`Google Gemini`](google_gemini.md), [`Google Gemma API`](google_gemma_api.md), [`Detections Filter`](detections_filter.md), [`VLM As Classifier`](vlm_as_classifier.md), [`PLC Reader`](plc_reader.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Moondream2`](moondream2.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`SIFT Comparison`](sift_comparison.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`EasyOCR`](easy_ocr.md), [`Google Gemini`](google_gemini.md), [`CogVLM`](cog_vlm.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Motion Detection`](motion_detection.md), [`SORT Tracker`](sort_tracker.md), [`OpenAI`](open_ai.md), [`Object Detection Model`](object_detection_model.md), [`Identify Outliers`](identify_outliers.md), [`Time in Zone`](timein_zone.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Time in Zone`](timein_zone.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`SIFT Comparison`](sift_comparison.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Image Stack`](image_stack.md), [`Line Counter`](line_counter.md), [`Google Gemma`](google_gemma.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Local File Sink`](local_file_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Object Detection Model`](object_detection_model.md), [`Detections Stitch`](detections_stitch.md), [`Byte Tracker`](byte_tracker.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`OCR Model`](ocr_model.md), [`Perspective Correction`](perspective_correction.md), [`Buffer`](buffer.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Identify Changes`](identify_changes.md), [`JSON Parser`](json_parser.md), [`Email Notification`](email_notification.md), [`Line Counter`](line_counter.md), [`S3 Sink`](s3_sink.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`LMM`](lmm.md), [`Distance Measurement`](distance_measurement.md), [`Detections Consensus`](detections_consensus.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Pixel Color Count`](pixel_color_count.md), [`Dimension Collapse`](dimension_collapse.md), [`Size Measurement`](size_measurement.md), [`SAM 3`](sam3.md), [`Time in Zone`](timein_zone.md), [`Detection Event Log`](detection_event_log.md), [`Event Writer`](event_writer.md), [`Image Contours`](image_contours.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Path Deviation`](path_deviation.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md)
    - outputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Triangle Visualization`](triangle_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Detections Combine`](detections_combine.md), [`Slack Notification`](slack_notification.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Florence-2 Model`](florence2_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Template Matching`](template_matching.md), [`Track Class Lock`](track_class_lock.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Halo Visualization`](halo_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Webhook Sink`](webhook_sink.md), [`SAM 3`](sam3.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`YOLO-World Model`](yolo_world_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Label Visualization`](label_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detection Offset`](detection_offset.md), [`Florence-2 Model`](florence2_model.md), [`Path Deviation`](path_deviation.md), [`Text Display`](text_display.md), [`MQTT Writer`](mqtt_writer.md), [`PLC Writer`](plc_writer.md), [`Gaze Detection`](gaze_detection.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Google Gemini`](google_gemini.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Polygon Visualization`](polygon_visualization.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Overlap Analysis`](overlap_analysis.md), [`Motion Detection`](motion_detection.md), [`Time in Zone`](timein_zone.md), [`Time in Zone`](timein_zone.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Image Slicer`](image_slicer.md), [`Polygon Visualization`](polygon_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Byte Tracker`](byte_tracker.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Slicer`](image_slicer.md), [`Identify Changes`](identify_changes.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Line Counter`](line_counter.md), [`Email Notification`](email_notification.md), [`Distance Measurement`](distance_measurement.md), [`Detections Consensus`](detections_consensus.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Size Measurement`](size_measurement.md), [`Trace Visualization`](trace_visualization.md), [`SAM 3`](sam3.md), [`Circle Visualization`](circle_visualization.md), [`Time in Zone`](timein_zone.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Event Writer`](event_writer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Path Deviation`](path_deviation.md), [`Byte Tracker`](byte_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Email Notification`](email_notification.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Detections Merge`](detections_merge.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Detections Transformation`](detections_transformation.md), [`Camera Focus`](camera_focus.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Dynamic Crop`](dynamic_crop.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Velocity`](velocity.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Object Detection Model`](object_detection_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Overlap Filter`](overlap_filter.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Detections Filter`](detections_filter.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Corner Visualization`](corner_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Identify Outliers`](identify_outliers.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Image Stack`](image_stack.md), [`Line Counter`](line_counter.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stitch Images`](stitch_images.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Keypoint Visualization`](keypoint_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Detections Stitch`](detections_stitch.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Perspective Correction`](perspective_correction.md), [`Crop Visualization`](crop_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Detection Event Log`](detection_event_log.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Detections Consensus` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `predictions_batches` (*Union[[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md)]*): List of references to detection predictions from multiple models. Each model's predictions must be made against the same input image. Predictions can be from object detection, instance segmentation, or keypoint detection models. The block matches overlapping detections across models and requires a minimum number of votes (required_votes) before accepting detections in the consensus output. Requires at least one prediction source. Supports batch processing..
        - `required_votes` (*[`integer`](../kinds/integer.md)*): Minimum number of votes (matching detections from different models) required to accept a detection in the consensus output. Detections that receive fewer votes than this threshold are filtered out. For example, if set to 2, at least 2 models must detect an overlapping object (above IoU threshold) for it to appear in the consensus results. Higher values create stricter consensus requirements, reducing false positives but potentially missing detections seen by fewer models..
        - `class_aware` (*[`boolean`](../kinds/boolean.md)*): If true, only detections with matching class names from different models are considered as votes for the same object. If false, any overlapping detections (regardless of class) contribute votes. Class-aware mode is more conservative and ensures class consistency in consensus, while class-agnostic mode allows voting across different classes but may merge detections of different object types..
        - `iou_threshold` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Intersection over Union (IoU) threshold for considering detections from different models as matching the same object. Detections with IoU above this threshold are considered overlapping and contribute votes to each other. Lower values (e.g., 0.2) are more lenient and match detections with less overlap, while higher values (e.g., 0.5) require stronger spatial overlap for matching. Typical values range from 0.2 to 0.5..
        - `confidence` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Confidence threshold applied to merged consensus detections. Only detections with aggregated confidence scores above this threshold are included in the output. Set to 0.0 to disable confidence filtering. Higher values filter out low-confidence consensus detections, improving output quality at the cost of potentially removing valid but lower-confidence detections..
        - `classes_to_consider` (*[`list_of_values`](../kinds/list_of_values.md)*): Optional list of class names to include in the consensus procedure. If provided, only detections of these classes are considered for voting and merging; all other classes are filtered out before consensus matching. Use this to focus consensus on specific object types while ignoring irrelevant detections. If None, all classes participate in consensus..
        - `required_objects` (*Union[[`dictionary`](../kinds/dictionary.md), [`integer`](../kinds/integer.md)]*): Optional minimum number of objects required to determine object presence. Can be an integer (total objects across all classes) or a dictionary mapping class names to per-class minimum counts. Used in conjunction with object_present output to determine if sufficient objects of each class are detected. For example, 3 means at least 3 total objects must be present, while {'person': 2, 'car': 1} requires at least 2 persons and 1 car. If None, object presence is determined solely by whether any consensus detections exist..

    - output
    
        - `predictions` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Prediction with detected bounding boxes in form of sv.Detections(...) object if `object_detection_prediction` or Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object if `instance_segmentation_prediction`.
        - `object_present` (*Union[[`boolean`](../kinds/boolean.md), [`dictionary`](../kinds/dictionary.md)]*): Boolean flag if `boolean` or Dictionary if `dictionary`.
        - `presence_confidence` (*Union[[`float_zero_to_one`](../kinds/float_zero_to_one.md), [`dictionary`](../kinds/dictionary.md)]*): `float` value in range `[0.0, 1.0]` if `float_zero_to_one` or Dictionary if `dictionary`.



??? tip "Example JSON definition of step `Detections Consensus` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/detections_consensus@v1",
	    "predictions_batches": [
	        "$steps.a.predictions",
	        "$steps.b.predictions"
	    ],
	    "required_votes": 2,
	    "class_aware": true,
	    "iou_threshold": 0.3,
	    "confidence": 0.1,
	    "classes_to_consider": [
	        "a",
	        "b"
	    ],
	    "required_objects": 3,
	    "presence_confidence_aggregation": "max",
	    "detections_merge_confidence_aggregation": "min",
	    "detections_merge_coordinates_aggregation": "min",
	    "detections_merge_mask_aggregation": "union"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    


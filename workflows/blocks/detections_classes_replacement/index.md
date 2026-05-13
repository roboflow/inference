
# Detections Classes Replacement



??? "Class: `DetectionsClassesReplacementBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/fusion/detections_classes_replacement/v1.py">inference.core.workflows.core_steps.fusion.detections_classes_replacement.v1.DetectionsClassesReplacementBlockV1</a>
    



Replace class labels of detection bounding boxes with classes predicted by a classification model applied to cropped regions, combining generic detection results with specialized classification predictions to enable two-stage detection workflows, fine-grained classification, and class refinement workflows where generic detections are refined with specific class labels from specialized classifiers.

## How This Block Works

This block combines results from a detection model (with bounding boxes and generic classes) with classification predictions (from a specialized classifier applied to cropped regions) to replace generic class labels with specific ones. The block:

1. Receives two inputs with different dimensionality levels:
   - `object_detection_predictions`: Detection results (dimensionality level 1) containing bounding boxes with generic classes (e.g., "dog", "person", "vehicle")
   - `classification_predictions`: Classification results (dimensionality level 2) from a classifier applied to cropped regions of each detection (e.g., "Golden Retriever", "Labrador" for dog detections). Can also be a list of strings (e.g. from OCR).
2. Matches classifications to detections:
   - Uses `PARENT_ID_KEY` (detection_id) in classification predictions to link each classification result to its source detection, OR
   - Uses positional mapping (order-based) if predictions are raw strings/lists without parent IDs.
3. Extracts leading class from each classification prediction:

   **For single-label classifications:**
   - Uses the "top" class (predicted class) from the classification result
   - Extracts class name, class ID, and confidence from the classification prediction

   **For multi-label classifications:**
   - Finds the class with the highest confidence score
   - Uses the most confident label as the replacement class
   - Extracts class name, class ID, and confidence from the highest-confidence prediction
   
   **For string predictions:**
   - Uses the string as the class name
   - Assigns a default confidence of 1.0 and class ID of 0

4. Handles missing classifications:
   - Detections without corresponding classification predictions are discarded by default
   - If `fallback_class_name` is provided, detections without classifications use the fallback class instead of being discarded
   - Fallback class ID is set to the provided value, or `sys.maxsize` if not specified or negative
5. Filters detections:
   - Keeps only detections that have classification results (or fallback if specified)
   - Removes detections that cannot be matched to classification predictions
6. Replaces class information:
   - Replaces class names in detections with classification class names
   - Replaces class IDs in detections with classification class IDs
   - Replaces confidence scores in detections with classification confidence scores
   - Updates all detection metadata to reflect the new class information
7. Generates new detection IDs:
   - Creates new unique detection IDs for updated detections (prevents ID conflicts)
   - Ensures detection IDs are unique after class replacement
8. Returns updated detections:
   - Outputs detections with replaced classes, maintaining bounding box coordinates and other properties
   - Output dimensionality matches input detection predictions (dimensionality level 1)

The block enables two-stage detection workflows where a generic detection model locates objects and a specialized classification model provides fine-grained labels. This is useful when you need generic localization (e.g., "dog") combined with specific classification (e.g., "Golden Retriever", "German Shepherd") without losing spatial information.

## Common Use Cases

- **Two-Stage Detection and Classification**: Combine generic detection with specialized classification for fine-grained labeling (e.g., detect "dog" then classify breed, detect "vehicle" then classify type, detect "person" then classify age group), enabling two-stage detection workflows
- **Class Refinement**: Refine generic class labels with specific classifications from specialized models (e.g., refine "animal" to specific species, refine "vehicle" to specific models, refine "food" to specific dishes), enabling class refinement workflows
- **Multi-Model Workflows**: Combine detection and classification models to leverage the strengths of both (e.g., use generic detector for localization and specialist classifier for identification, combine coarse and fine-grained models, leverage specialized classifiers with general detectors), enabling multi-model workflows
- **Hierarchical Classification**: Apply hierarchical classification where detection provides high-level classes and classification provides detailed sub-classes (e.g., detect "mammal" then classify species, detect "plant" then classify variety, detect "structure" then classify type), enabling hierarchical classification workflows
- **Crop-Based Classification**: Use classification results from cropped regions to enhance detection results (e.g., classify crops to improve detection labels, apply specialized classifiers to detected regions, refine detections with crop classifications), enabling crop-based classification workflows
- **Fine-Grained Object Recognition**: Enable fine-grained recognition by combining localization and detailed classification (e.g., recognize specific product models, identify specific animal breeds, classify specific vehicle types), enabling fine-grained recognition workflows

## Connecting to Other Blocks

This block receives detection and classification predictions and produces detections with replaced classes:

- **After detection and classification model blocks** to combine generic detection with specialized classification (e.g., object detection + classification to refined detections, detection model + classifier to labeled detections), enabling detection-classification fusion workflows
- **After crop blocks** that create crops from detections for classification (e.g., crop detections then classify crops, create crops for classification then replace classes), enabling crop-classification workflows
- **Before visualization blocks** to display detections with refined classes (e.g., visualize refined detections, display detections with specific labels, show classification-enhanced detections), enabling refined detection visualization workflows
- **Before filtering blocks** to filter detections with refined classes (e.g., filter by specific classes, filter refined detections, apply filters to classified detections), enabling refined detection filtering workflows
- **Before analytics blocks** to perform analytics on refined detections (e.g., analyze specific classes, perform analytics on classified detections, track refined detection metrics), enabling refined detection analytics workflows
- **In workflow outputs** to provide refined detections as final output (e.g., two-stage detection outputs, classification-enhanced detection outputs, refined detection results), enabling refined detection output workflows

## Requirements

This block requires object detection predictions (with bounding boxes) and classification predictions from crops of those bounding boxes. The classification predictions must have `PARENT_ID_KEY` (detection_id) to link classifications to their source detections. The block accepts different dimensionality levels: detection predictions at level 1 and classification predictions at level 2 (from crops). For single-label classifications, the "top" class is used. For multi-label classifications, the most confident class is selected. Detections without classification results are discarded unless `fallback_class_name` is provided. The block outputs detections with replaced classes, class IDs, and confidences, with new detection IDs generated. Output dimensionality matches input detection predictions (level 1).


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/detections_classes_replacement@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `fallback_class_name` | `str` | Optional class name to use for detections that don't have corresponding classification predictions. If not provided (default None), detections without classifications are discarded. If provided, detections without classifications use this fallback class name instead of being removed. Useful for preserving detections when classification fails or is unavailable.. | ✅ |
| `fallback_class_id` | `int` | Optional class ID to use with fallback_class_name for detections without classification predictions. If not specified or negative, the class ID is set to sys.maxsize. Only used when fallback_class_name is provided. Should match the class ID mapping used in your model.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Detections Classes Replacement` in version `v1`.

    - inputs: [`Slack Notification`](slack_notification.md), [`Camera Focus`](camera_focus.md), [`Time in Zone`](timein_zone.md), [`Perspective Correction`](perspective_correction.md), [`Detections Combine`](detections_combine.md), [`SAM 3`](sam3.md), [`Dimension Collapse`](dimension_collapse.md), [`EasyOCR`](easy_ocr.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Detection Offset`](detection_offset.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`S3 Sink`](s3_sink.md), [`SIFT Comparison`](sift_comparison.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`CogVLM`](cog_vlm.md), [`Detection Event Log`](detection_event_log.md), [`Velocity`](velocity.md), [`SIFT Comparison`](sift_comparison.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Detections Transformation`](detections_transformation.md), [`Byte Tracker`](byte_tracker.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`Moondream2`](moondream2.md), [`Line Counter`](line_counter.md), [`Google Gemma API`](google_gemma_api.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`OpenAI`](open_ai.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`LMM For Classification`](lmm_for_classification.md), [`Dynamic Zone`](dynamic_zone.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Clip Comparison`](clip_comparison.md), [`Anthropic Claude`](anthropic_claude.md), [`CSV Formatter`](csv_formatter.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Time in Zone`](timein_zone.md), [`Object Detection Model`](object_detection_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`OCR Model`](ocr_model.md), [`Line Counter`](line_counter.md), [`SORT Tracker`](sort_tracker.md), [`OpenAI`](open_ai.md), [`Detections Consensus`](detections_consensus.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`SAM 3`](sam3.md), [`Seg Preview`](seg_preview.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Vision OCR`](google_vision_ocr.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Path Deviation`](path_deviation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Motion Detection`](motion_detection.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Dynamic Crop`](dynamic_crop.md), [`Byte Tracker`](byte_tracker.md), [`Local File Sink`](local_file_sink.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Contours`](image_contours.md), [`VLM As Detector`](vlm_as_detector.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Overlap Filter`](overlap_filter.md), [`Anthropic Claude`](anthropic_claude.md), [`SAM 3`](sam3.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Template Matching`](template_matching.md), [`Detections Merge`](detections_merge.md), [`Anthropic Claude`](anthropic_claude.md), [`Clip Comparison`](clip_comparison.md), [`YOLO-World Model`](yolo_world_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Email Notification`](email_notification.md), [`Detections Stitch`](detections_stitch.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Distance Measurement`](distance_measurement.md), [`Byte Tracker`](byte_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Size Measurement`](size_measurement.md), [`Time in Zone`](timein_zone.md), [`Path Deviation`](path_deviation.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Email Notification`](email_notification.md), [`Florence-2 Model`](florence2_model.md), [`Detections Filter`](detections_filter.md), [`Webhook Sink`](webhook_sink.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`GLM-OCR`](glmocr.md), [`Pixel Color Count`](pixel_color_count.md), [`OpenAI`](open_ai.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Google Gemini`](google_gemini.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemini`](google_gemini.md), [`Mask Area Measurement`](mask_area_measurement.md), [`LMM`](lmm.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Google Gemini`](google_gemini.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Gaze Detection`](gaze_detection.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Buffer`](buffer.md), [`Qwen 3.6 API`](qwen3.6_api.md)
    - outputs: [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Camera Focus`](camera_focus.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Time in Zone`](timein_zone.md), [`Perspective Correction`](perspective_correction.md), [`Crop Visualization`](crop_visualization.md), [`Detections Combine`](detections_combine.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Label Visualization`](label_visualization.md), [`Path Deviation`](path_deviation.md), [`Corner Visualization`](corner_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Byte Tracker`](byte_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Detection Offset`](detection_offset.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Overlap Filter`](overlap_filter.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Triangle Visualization`](triangle_visualization.md), [`Detections Merge`](detections_merge.md), [`Dot Visualization`](dot_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Velocity`](velocity.md), [`Distance Measurement`](distance_measurement.md), [`Byte Tracker`](byte_tracker.md), [`Trace Visualization`](trace_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Size Measurement`](size_measurement.md), [`Icon Visualization`](icon_visualization.md), [`Time in Zone`](timein_zone.md), [`Path Deviation`](path_deviation.md), [`Halo Visualization`](halo_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Byte Tracker`](byte_tracker.md), [`Color Visualization`](color_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Detections Filter`](detections_filter.md), [`Florence-2 Model`](florence2_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Line Counter`](line_counter.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Polygon Visualization`](polygon_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Polygon Visualization`](polygon_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Circle Visualization`](circle_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Time in Zone`](timein_zone.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Line Counter`](line_counter.md), [`SORT Tracker`](sort_tracker.md), [`Detections Consensus`](detections_consensus.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Detections Classes Replacement` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `object_detection_predictions` (*Union[[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md)]*): Detection predictions (object detection, instance segmentation, or keypoint detection) containing bounding boxes with generic class labels that will be replaced with classification results. These detections should correspond to the regions that were cropped and classified. Detections must have detection IDs that match the PARENT_ID_KEY in classification predictions. Detections at dimensionality level 1..
        - `classification_predictions` (*Union[[`list_of_values`](../kinds/list_of_values.md), [`classification_prediction`](../kinds/classification_prediction.md), [`string`](../kinds/string.md)]*): Labels to replace detection class names with. Accepts classification predictions (linked via parent_id), plain strings, or lists of strings (e.g. OCR/LMM output like Gemini). String inputs are matched to detections positionally (1:1 by index). Classification inputs support single-label ('top' class) and multi-label (most confident class)..
        - `fallback_class_name` (*[`string`](../kinds/string.md)*): Optional class name to use for detections that don't have corresponding classification predictions. If not provided (default None), detections without classifications are discarded. If provided, detections without classifications use this fallback class name instead of being removed. Useful for preserving detections when classification fails or is unavailable..
        - `fallback_class_id` (*[`integer`](../kinds/integer.md)*): Optional class ID to use with fallback_class_name for detections without classification predictions. If not specified or negative, the class ID is set to sys.maxsize. Only used when fallback_class_name is provided. Should match the class ID mapping used in your model..

    - output
    
        - `predictions` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md)]*): Prediction with detected bounding boxes in form of sv.Detections(...) object if `object_detection_prediction` or Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object if `instance_segmentation_prediction` or Prediction with detected bounding boxes and detected keypoints in form of sv.Detections(...) object if `keypoint_detection_prediction`.



??? tip "Example JSON definition of step `Detections Classes Replacement` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/detections_classes_replacement@v1",
	    "object_detection_predictions": "$steps.object_detection_model.predictions",
	    "classification_predictions": "$steps.classification_model.predictions",
	    "fallback_class_name": null,
	    "fallback_class_id": null
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    


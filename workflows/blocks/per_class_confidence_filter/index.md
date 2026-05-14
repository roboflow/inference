
# Per-Class Confidence Filter



??? "Class: `PerClassConfidenceFilterBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/transformations/per_class_confidence_filter/v1.py">inference.core.workflows.core_steps.transformations.per_class_confidence_filter.v1.PerClassConfidenceFilterBlockV1</a>
    



Filter detection predictions by applying a different confidence threshold to each class, keeping only detections whose confidence meets or exceeds the threshold configured for their class (with a configurable fallback threshold for classes that are not listed).

## How This Block Works

This block applies class-aware confidence filtering to detection predictions, enabling precise control over which detections are retained based on per-class quality requirements. The block:

1. Takes detection predictions (object detection, instance segmentation, or keypoint detection) and a dictionary mapping class names to confidence thresholds
2. Iterates through each detection, looking up the threshold associated with the detection's class name
3. If the class is not present in the dictionary, falls back to the configurable `default_threshold` value
4. Keeps only the detections whose confidence is greater than or equal to the resolved threshold
5. Returns the filtered detections while preserving all original metadata (class ids, masks, keypoints, tracker ids, etc.)

Unlike a single global confidence threshold, this block lets you demand high-confidence predictions for classes that are prone to false positives while keeping a more permissive threshold for classes that are harder to detect. Unlike the generic detections filter, it exposes a purpose-built dictionary input that maps cleanly to a simple `{"class_name": threshold}` JSON object.

## Common Use Cases

- **Noise-prone classes**: Demand very high confidence (e.g. 0.9) for classes that frequently produce false positives, while accepting lower confidence for well-behaved classes
- **Hard-to-detect classes**: Lower the threshold for classes that the model rarely detects with high confidence so that they are not filtered out entirely
- **Production-grade filtering**: Apply domain-specific thresholds tuned during evaluation so that downstream analytics, alerts, or counting blocks only see detections that meet the project's quality bar
- **Multi-class pipelines**: Combine with object detection models that predict many classes at once when a single global confidence threshold is too coarse

## Connecting to Other Blocks

The filtered predictions from this block can be connected to:

- **Visualization blocks** (Bounding Box Visualization, Label Visualization, Polygon Visualization) to render only detections that cleared their per-class threshold
- **Counting and analytics blocks** (Line Counter, Time in Zone, Velocity) so that metrics reflect only high-quality detections
- **Tracking blocks** (Byte Tracker) so that tracker associations are not polluted by low-confidence noise
- **Storage or sink blocks** (Roboflow Dataset Upload, Webhook Sink, CSV Formatter) so that only detections meeting the quality bar are persisted or transmitted
- **Downstream transformation blocks** (Dynamic Crop, Detection Offset) for subsequent processing on the filtered subset


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/per_class_confidence_filter@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `class_thresholds` | `Dict[str, float]` | Mapping of class name to minimum confidence threshold. Detections whose class name is present in this dictionary are kept only if their confidence is at least the corresponding threshold. Classes not present fall back to default_threshold. Thresholds should be in the [0.0, 1.0] range.. | ✅ |
| `default_threshold` | `float` | Confidence threshold applied to detections whose class name is not listed in class_thresholds. Must be in the [0.0, 1.0] range.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Per-Class Confidence Filter` in version `v1`.

    - inputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`Time in Zone`](timein_zone.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`SAM 3`](sam3.md), [`Seg Preview`](seg_preview.md), [`Detections Transformation`](detections_transformation.md), [`Line Counter`](line_counter.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Dynamic Zone`](dynamic_zone.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Florence-2 Model`](florence2_model.md), [`SORT Tracker`](sort_tracker.md), [`Overlap Filter`](overlap_filter.md), [`Gaze Detection`](gaze_detection.md), [`Velocity`](velocity.md), [`OpenAI`](open_ai.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Google Vision OCR`](google_vision_ocr.md), [`Object Detection Model`](object_detection_model.md), [`Detections Combine`](detections_combine.md), [`Time in Zone`](timein_zone.md), [`Florence-2 Model`](florence2_model.md), [`Path Deviation`](path_deviation.md), [`Motion Detection`](motion_detection.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Qwen3.5`](qwen3.5.md), [`SAM 3`](sam3.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Clip Comparison`](clip_comparison.md), [`YOLO-World Model`](yolo_world_model.md), [`Detections Stitch`](detections_stitch.md), [`Byte Tracker`](byte_tracker.md), [`Bounding Rectangle`](bounding_rectangle.md), [`SmolVLM2`](smol_vlm2.md), [`Identify Changes`](identify_changes.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Dynamic Crop`](dynamic_crop.md), [`Byte Tracker`](byte_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Object Detection Model`](object_detection_model.md), [`Detections Consensus`](detections_consensus.md), [`Time in Zone`](timein_zone.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detections Merge`](detections_merge.md), [`VLM As Detector`](vlm_as_detector.md), [`Detection Offset`](detection_offset.md), [`SAM 3`](sam3.md), [`OCR Model`](ocr_model.md), [`EasyOCR`](easy_ocr.md), [`Qwen3-VL`](qwen3_vl.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Path Deviation`](path_deviation.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Moondream2`](moondream2.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Byte Tracker`](byte_tracker.md), [`CogVLM`](cog_vlm.md), [`Template Matching`](template_matching.md), [`Detections Filter`](detections_filter.md), [`LMM`](lmm.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Identify Outliers`](identify_outliers.md), [`Perspective Correction`](perspective_correction.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Object Detection Model`](object_detection_model.md)
    - outputs: [`Triangle Visualization`](triangle_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Time in Zone`](timein_zone.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Dynamic Zone`](dynamic_zone.md), [`Florence-2 Model`](florence2_model.md), [`SORT Tracker`](sort_tracker.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Detections Combine`](detections_combine.md), [`Path Deviation`](path_deviation.md), [`Distance Measurement`](distance_measurement.md), [`Byte Tracker`](byte_tracker.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Camera Focus`](camera_focus.md), [`Detection Event Log`](detection_event_log.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Line Counter`](line_counter.md), [`Crop Visualization`](crop_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Perspective Correction`](perspective_correction.md), [`Halo Visualization`](halo_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Line Counter`](line_counter.md), [`Mask Edge Snap`](mask_edge_snap.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Icon Visualization`](icon_visualization.md), [`Overlap Filter`](overlap_filter.md), [`Circle Visualization`](circle_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Velocity`](velocity.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Time in Zone`](timein_zone.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Florence-2 Model`](florence2_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Detections Stitch`](detections_stitch.md), [`Halo Visualization`](halo_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Dynamic Crop`](dynamic_crop.md), [`Color Visualization`](color_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Label Visualization`](label_visualization.md), [`Time in Zone`](timein_zone.md), [`Detections Consensus`](detections_consensus.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detections Merge`](detections_merge.md), [`Dot Visualization`](dot_visualization.md), [`Size Measurement`](size_measurement.md), [`Detection Offset`](detection_offset.md), [`Trace Visualization`](trace_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Path Deviation`](path_deviation.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Byte Tracker`](byte_tracker.md), [`Detections Filter`](detections_filter.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Bounding Rectangle`](bounding_rectangle.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Per-Class Confidence Filter` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `predictions` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md)]*): Detection predictions to filter. Each detection is kept only if its confidence is greater than or equal to the threshold configured for its class (with a fallback to default_threshold for classes that are not listed in class_thresholds)..
        - `class_thresholds` (*[`dictionary`](../kinds/dictionary.md)*): Mapping of class name to minimum confidence threshold. Detections whose class name is present in this dictionary are kept only if their confidence is at least the corresponding threshold. Classes not present fall back to default_threshold. Thresholds should be in the [0.0, 1.0] range..
        - `default_threshold` (*[`float_zero_to_one`](../kinds/float_zero_to_one.md)*): Confidence threshold applied to detections whose class name is not listed in class_thresholds. Must be in the [0.0, 1.0] range..

    - output
    
        - `predictions` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md)]*): Prediction with detected bounding boxes in form of sv.Detections(...) object if `object_detection_prediction` or Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object if `instance_segmentation_prediction` or Prediction with detected bounding boxes and detected keypoints in form of sv.Detections(...) object if `keypoint_detection_prediction`.



??? tip "Example JSON definition of step `Per-Class Confidence Filter` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/per_class_confidence_filter@v1",
	    "predictions": "$steps.object_detection_model.predictions",
	    "class_thresholds": {
	        "car": 0.5,
	        "person": 0.98
	    },
	    "default_threshold": 0.3
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    


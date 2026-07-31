
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

    - inputs: [`Track Class Lock`](track_class_lock.md), [`Byte Tracker`](byte_tracker.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Path Deviation`](path_deviation.md), [`VLM As Detector`](vlm_as_detector.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Detections Stitch`](detections_stitch.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`CogVLM`](cog_vlm.md), [`Time in Zone`](timein_zone.md), [`Template Matching`](template_matching.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Perspective Correction`](perspective_correction.md), [`Motion Detection`](motion_detection.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`YOLO-World Model`](yolo_world_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Overlap Filter`](overlap_filter.md), [`Dynamic Crop`](dynamic_crop.md), [`PLC Reader`](plc_reader.md), [`Detections Consensus`](detections_consensus.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OCR Model`](ocr_model.md), [`GeoTag Detection`](geo_tag_detection.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Time in Zone`](timein_zone.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Velocity`](velocity.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Time in Zone`](timein_zone.md), [`Detection Event Log`](detection_event_log.md), [`Path Deviation`](path_deviation.md), [`LMM`](lmm.md), [`Detections Combine`](detections_combine.md), [`Detection Offset`](detection_offset.md), [`SAM 3`](sam3.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Seg Preview`](seg_preview.md), [`EasyOCR`](easy_ocr.md), [`Florence-2 Model`](florence2_model.md), [`Qwen3.5`](qwen3.5.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`PP-OCR`](ppocr.md), [`Object Detection Model`](object_detection_model.md), [`SAM 3`](sam3.md), [`Line Counter`](line_counter.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Dynamic Zone`](dynamic_zone.md), [`Detections Merge`](detections_merge.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Google Vision OCR`](google_vision_ocr.md), [`Detections Filter`](detections_filter.md), [`SAM 3`](sam3.md), [`Detections Transformation`](detections_transformation.md), [`SORT Tracker`](sort_tracker.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Identify Changes`](identify_changes.md), [`Qwen3-VL`](qwen3_vl.md), [`Gaze Detection`](gaze_detection.md), [`Object Detection Model`](object_detection_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Florence-2 Model`](florence2_model.md), [`Identify Outliers`](identify_outliers.md), [`Moondream2`](moondream2.md), [`Nearest Neighbor Detection Match`](nearest_neighbor_detection_match.md), [`Byte Tracker`](byte_tracker.md), [`SmolVLM2`](smol_vlm2.md), [`Byte Tracker`](byte_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`VLM As Detector`](vlm_as_detector.md)
    - outputs: [`Track Class Lock`](track_class_lock.md), [`Byte Tracker`](byte_tracker.md), [`Blur Visualization`](blur_visualization.md), [`Frame Delay`](frame_delay.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Path Deviation`](path_deviation.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Crop Visualization`](crop_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Event Writer`](event_writer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Time in Zone`](timein_zone.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Perspective Correction`](perspective_correction.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Halo Visualization`](halo_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Label Visualization`](label_visualization.md), [`Overlap Filter`](overlap_filter.md), [`Dynamic Crop`](dynamic_crop.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Detections Consensus`](detections_consensus.md), [`Label Visualization`](label_visualization.md), [`GeoTag Detection`](geo_tag_detection.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Halo Visualization`](halo_visualization.md), [`Time in Zone`](timein_zone.md), [`Velocity`](velocity.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Time in Zone`](timein_zone.md), [`Detection Event Log`](detection_event_log.md), [`Overlap Analysis`](overlap_analysis.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Size Measurement`](size_measurement.md), [`Path Deviation`](path_deviation.md), [`Corner Visualization`](corner_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Detections Combine`](detections_combine.md), [`Detection Offset`](detection_offset.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Distance Measurement`](distance_measurement.md), [`Dot Visualization`](dot_visualization.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Florence-2 Model`](florence2_model.md), [`Background Color Visualization`](background_color_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Trace Visualization`](trace_visualization.md), [`Color Visualization`](color_visualization.md), [`Camera Focus`](camera_focus.md), [`Line Counter`](line_counter.md), [`Circle Visualization`](circle_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Detections Merge`](detections_merge.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Detections Filter`](detections_filter.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Rich Label Visualization`](rich_label_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Line Counter`](line_counter.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Mask Visualization`](mask_visualization.md), [`Nearest Neighbor Detection Match`](nearest_neighbor_detection_match.md), [`Byte Tracker`](byte_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Pixelate Visualization`](pixelate_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Per-Class Confidence Filter` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `predictions` (*Union[[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md)]*): Detection predictions to filter. Each detection is kept only if its confidence is greater than or equal to the threshold configured for its class (with a fallback to default_threshold for classes that are not listed in class_thresholds)..
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


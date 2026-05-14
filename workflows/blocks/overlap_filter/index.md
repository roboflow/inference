
# Overlap Filter



??? "Class: `OverlapBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/analytics/overlap/v1.py">inference.core.workflows.core_steps.analytics.overlap.v1.OverlapBlockV1</a>
    



Filter detection predictions to keep only objects that overlap with instances of a specified class, enabling spatial relationship filtering to identify objects that are positioned relative to other objects (e.g., people on bicycles, items on pallets, objects in containers).

## How This Block Works

This block filters detections based on spatial overlap relationships with a specified overlap class. The block:

1. Takes detection predictions (object detection or instance segmentation) and an overlap class name as input
2. Separates detections into two groups:
   - **Overlap class detections**: Objects matching the specified `overlap_class_name` (e.g., "bicycle", "pallet", "car")
   - **Other detections**: All remaining objects that may overlap with the overlap class
3. For each overlap class detection, identifies other detections that spatially overlap with it using one of two overlap modes:
   - **Center Overlap**: Checks if the center point of other detections falls within the overlap class bounding box (more precise, requires the center to be inside)
   - **Any Overlap**: Checks if there's any spatial intersection between bounding boxes (more lenient, any overlap counts)
4. Collects all detections that overlap with any overlap class instance
5. Filters out the overlap class detections themselves from the output
6. Returns only the overlapping detections (objects that are positioned relative to the overlap class)

The block effectively answers: "Which objects are overlapping with instances of class X?" For example, if you specify "bicycle" as the overlap class, the block finds people or other objects that overlap with bicycles, but removes the bicycles themselves from the output. This enables workflows to identify objects that have spatial relationships with specific reference classes, such as identifying items on surfaces, objects in containers, or people on vehicles.

## Common Use Cases

- **Person-on-Vehicle Detection**: Identify people on bicycles, motorcycles, or other vehicles by using the vehicle class as the overlap class (e.g., filter for people overlapping with "bicycle" detections), enabling detection of riders, passengers, or people using vehicles
- **Items on Surfaces**: Find objects positioned on pallets, tables, or shelves by using the surface class as the overlap class (e.g., filter for items overlapping with "pallet" detections), enabling inventory tracking, object counting on surfaces, or surface occupancy analysis
- **Objects in Containers**: Identify items inside containers, boxes, or vehicles by using the container class as the overlap class (e.g., filter for objects overlapping with "container" detections), enabling content detection, loading verification, or container monitoring
- **Spatial Relationship Filtering**: Filter detections based on proximity or containment relationships (e.g., find all objects that are inside or overlapping with a specific class), enabling conditional processing based on spatial arrangements
- **Nested Object Detection**: Identify objects that are part of or attached to other objects (e.g., find equipment attached to vehicles, accessories on people), enabling detection of composite objects or object relationships
- **Zone-Based Filtering**: Use overlap class as a reference zone to find objects that intersect with specific regions (e.g., filter objects overlapping with "parking_space" class), enabling zone-based analysis and conditional detection filtering

## Connecting to Other Blocks

The filtered overlapping detections from this block can be connected to:

- **Detection model blocks** (e.g., Object Detection Model, Instance Segmentation Model) to receive predictions that are filtered to show only objects overlapping with a specified reference class, enabling spatial relationship analysis
- **Visualization blocks** (e.g., Bounding Box Visualization, Polygon Visualization, Label Visualization) to display only the overlapping objects, highlighting objects that have spatial relationships with the reference class
- **Counting and analytics blocks** (e.g., Line Counter, Time in Zone, Velocity) to count or analyze only overlapping objects (e.g., count people on bicycles, track items on pallets), providing metrics for spatially related objects
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload, Webhook Sink) to save or transmit filtered overlapping detection results, storing data about objects with specific spatial relationships
- **Filtering blocks** (e.g., Detections Filter) to apply additional filtering criteria to the overlapping detections, enabling multi-stage filtering workflows
- **Flow control blocks** (e.g., Continue If) to conditionally trigger downstream processing based on whether overlapping objects are detected, enabling workflows that respond to spatial relationships


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/overlap@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `overlap_type` | `str` | Method for determining spatial overlap between detections. 'Center Overlap' checks if the center point of other detections falls within the overlap class bounding box (more precise, requires center to be inside). 'Any Overlap' checks if there's any spatial intersection between bounding boxes (more lenient, any overlap counts). Center Overlap is stricter and better for containment relationships, while Any Overlap is more inclusive and better for detecting any proximity or partial overlap.. | ❌ |
| `overlap_class_name` | `str` | Class name of the reference objects used for overlap detection. Detections matching this class name are used as reference points, and other detections that overlap with these reference objects are kept in the output. The overlap class detections themselves are removed from the results. Example: Use 'bicycle' to find people or objects overlapping with bicycles; use 'pallet' to find items on pallets.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Overlap Filter` in version `v1`.

    - inputs: [`Time in Zone`](timein_zone.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`SAM 3`](sam3.md), [`Seg Preview`](seg_preview.md), [`Detections Transformation`](detections_transformation.md), [`Line Counter`](line_counter.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Dynamic Zone`](dynamic_zone.md), [`Mask Edge Snap`](mask_edge_snap.md), [`SORT Tracker`](sort_tracker.md), [`Overlap Filter`](overlap_filter.md), [`Velocity`](velocity.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Google Vision OCR`](google_vision_ocr.md), [`Object Detection Model`](object_detection_model.md), [`Detections Combine`](detections_combine.md), [`Time in Zone`](timein_zone.md), [`Path Deviation`](path_deviation.md), [`Motion Detection`](motion_detection.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`SAM 3`](sam3.md), [`YOLO-World Model`](yolo_world_model.md), [`Detections Stitch`](detections_stitch.md), [`Byte Tracker`](byte_tracker.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Dynamic Crop`](dynamic_crop.md), [`Byte Tracker`](byte_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Object Detection Model`](object_detection_model.md), [`Detections Consensus`](detections_consensus.md), [`Time in Zone`](timein_zone.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detections Merge`](detections_merge.md), [`VLM As Detector`](vlm_as_detector.md), [`Detection Offset`](detection_offset.md), [`SAM 3`](sam3.md), [`OCR Model`](ocr_model.md), [`EasyOCR`](easy_ocr.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Path Deviation`](path_deviation.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Moondream2`](moondream2.md), [`Byte Tracker`](byte_tracker.md), [`Template Matching`](template_matching.md), [`Detections Filter`](detections_filter.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Perspective Correction`](perspective_correction.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Object Detection Model`](object_detection_model.md)
    - outputs: [`Triangle Visualization`](triangle_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Heatmap Visualization`](heatmap_visualization.md), [`Time in Zone`](timein_zone.md), [`Detections Transformation`](detections_transformation.md), [`Line Counter`](line_counter.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Florence-2 Model`](florence2_model.md), [`SORT Tracker`](sort_tracker.md), [`Icon Visualization`](icon_visualization.md), [`Overlap Filter`](overlap_filter.md), [`Circle Visualization`](circle_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Velocity`](velocity.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Detections Combine`](detections_combine.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Florence-2 Model`](florence2_model.md), [`Path Deviation`](path_deviation.md), [`Distance Measurement`](distance_measurement.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Time in Zone`](timein_zone.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Detections Stitch`](detections_stitch.md), [`Byte Tracker`](byte_tracker.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Camera Focus`](camera_focus.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Dynamic Crop`](dynamic_crop.md), [`Color Visualization`](color_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Label Visualization`](label_visualization.md), [`Detections Consensus`](detections_consensus.md), [`Time in Zone`](timein_zone.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detections Merge`](detections_merge.md), [`Dot Visualization`](dot_visualization.md), [`Size Measurement`](size_measurement.md), [`Detection Offset`](detection_offset.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Trace Visualization`](trace_visualization.md), [`Line Counter`](line_counter.md), [`Crop Visualization`](crop_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Path Deviation`](path_deviation.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Byte Tracker`](byte_tracker.md), [`Detections Filter`](detections_filter.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Perspective Correction`](perspective_correction.md), [`Byte Tracker`](byte_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Overlap Filter` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `predictions` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Detection predictions (object detection or instance segmentation) containing objects that may overlap with the specified overlap class. The block identifies detections matching the overlap_class_name and finds other detections that spatially overlap with them. Only the overlapping detections (not the overlap class itself) are returned in the output..

    - output
    
        - `overlaps` ([`object_detection_prediction`](../kinds/object_detection_prediction.md)): Prediction with detected bounding boxes in form of sv.Detections(...) object.



??? tip "Example JSON definition of step `Overlap Filter` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/overlap@v1",
	    "predictions": "$steps.object_detection_model.predictions",
	    "overlap_type": "Center Overlap",
	    "overlap_class_name": "<block_does_not_provide_example>"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    


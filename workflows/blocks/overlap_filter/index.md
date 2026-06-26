
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

    - inputs: [`YOLO-World Model`](yolo_world_model.md), [`Detection Event Log`](detection_event_log.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Detections Combine`](detections_combine.md), [`SAM 3`](sam3.md), [`Dynamic Zone`](dynamic_zone.md), [`Byte Tracker`](byte_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`SORT Tracker`](sort_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Google Vision OCR`](google_vision_ocr.md), [`Line Counter`](line_counter.md), [`Overlap Filter`](overlap_filter.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Detections Consensus`](detections_consensus.md), [`Time in Zone`](timein_zone.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`EasyOCR`](easy_ocr.md), [`Template Matching`](template_matching.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`Detections Stitch`](detections_stitch.md), [`Seg Preview`](seg_preview.md), [`OCR Model`](ocr_model.md), [`Byte Tracker`](byte_tracker.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Object Detection Model`](object_detection_model.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Detections Merge`](detections_merge.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Path Deviation`](path_deviation.md), [`Time in Zone`](timein_zone.md), [`Detection Offset`](detection_offset.md), [`VLM As Detector`](vlm_as_detector.md), [`Track Class Lock`](track_class_lock.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Perspective Correction`](perspective_correction.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Velocity`](velocity.md), [`Dynamic Crop`](dynamic_crop.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`SAM 3`](sam3.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Mask Area Measurement`](mask_area_measurement.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Detections Filter`](detections_filter.md), [`VLM As Detector`](vlm_as_detector.md), [`Detections Transformation`](detections_transformation.md), [`Path Deviation`](path_deviation.md), [`Moondream2`](moondream2.md), [`Motion Detection`](motion_detection.md), [`SAM 3`](sam3.md)
    - outputs: [`Florence-2 Model`](florence2_model.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`SORT Tracker`](sort_tracker.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Byte Tracker`](byte_tracker.md), [`Detections Consensus`](detections_consensus.md), [`Time in Zone`](timein_zone.md), [`Time in Zone`](timein_zone.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Byte Tracker`](byte_tracker.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Time in Zone`](timein_zone.md), [`Label Visualization`](label_visualization.md), [`Track Class Lock`](track_class_lock.md), [`Velocity`](velocity.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Corner Visualization`](corner_visualization.md), [`Distance Measurement`](distance_measurement.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Triangle Visualization`](triangle_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Path Deviation`](path_deviation.md), [`Detection Event Log`](detection_event_log.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Detections Combine`](detections_combine.md), [`Byte Tracker`](byte_tracker.md), [`Line Counter`](line_counter.md), [`Size Measurement`](size_measurement.md), [`Overlap Filter`](overlap_filter.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Line Counter`](line_counter.md), [`Circle Visualization`](circle_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Event Writer`](event_writer.md), [`Detections Merge`](detections_merge.md), [`Path Deviation`](path_deviation.md), [`Detection Offset`](detection_offset.md), [`Dot Visualization`](dot_visualization.md), [`Camera Focus`](camera_focus.md), [`Perspective Correction`](perspective_correction.md), [`Blur Visualization`](blur_visualization.md), [`Overlap Analysis`](overlap_analysis.md), [`Dynamic Crop`](dynamic_crop.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Color Visualization`](color_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Detections Filter`](detections_filter.md), [`Icon Visualization`](icon_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Background Color Visualization`](background_color_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Ellipse Visualization`](ellipse_visualization.md), [`SAM2 Video Tracker`](sam2_video_tracker.md)

    
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


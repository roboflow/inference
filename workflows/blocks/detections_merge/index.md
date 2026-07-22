
# Detections Merge



??? "Class: `DetectionsMergeBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/transformations/detections_merge/v1.py">inference.core.workflows.core_steps.transformations.detections_merge.v1.DetectionsMergeBlockV1</a>
    



Combine multiple detection predictions into a single merged detection with a union bounding box that encompasses all input detections, simplifying multiple detections into one larger detection region for overlapping object consolidation, region creation from multiple objects, and detection simplification workflows.

## How This Block Works

This block merges multiple detections into a single detection by calculating a union bounding box that contains all input detections. The block:

1. Receives detection predictions (object detection, instance segmentation, or keypoint detection) containing multiple detections
2. Validates input (handles empty detections by returning an empty detection result)
3. Calculates the union bounding box from all input detections:
   - Extracts all bounding box coordinates (xyxy format) from input detections
   - Finds the minimum x and y coordinates (leftmost and topmost points) across all boxes
   - Finds the maximum x and y coordinates (rightmost and bottommost points) across all boxes
   - Creates a single bounding box that completely encompasses all input detections
4. Determines the merged detection's confidence:
   - Finds the detection with the lowest confidence score among all input detections
   - Uses this lowest confidence as the merged detection's confidence (conservative approach)
   - Handles cases where confidence scores may not be present
5. Creates a new merged detection with:
   - The calculated union bounding box (encompasses all input detections)
   - A customizable class name (default: "merged_detection", configurable via class_name parameter)
   - The lowest confidence from input detections (conservative confidence assignment)
   - A fixed class_id of 0 for the merged detection
   - A newly generated detection ID (unique identifier for the merged detection)
6. Returns the single merged detection containing all input detections within its bounding box

The block creates a unified bounding box representation of multiple detections, useful for consolidating overlapping or nearby detections into a single region. The union bounding box approach ensures all original detections are completely contained within the merged detection. By using the lowest confidence, the block adopts a conservative approach, ensuring the merged detection's confidence reflects the least certain input detection. The merged detection can be customized with a class name to indicate its merged nature or to represent a specific category.

## Common Use Cases

- **Overlapping Detection Consolidation**: Merge multiple overlapping detections of the same or related objects into a single unified detection (e.g., merge overlapping detections of the same person from multiple frames, consolidate duplicate detections from different models, combine overlapping object parts into one detection), enabling overlapping detection simplification
- **Multi-Object Region Creation**: Create a single bounding box region that encompasses multiple detected objects for area-based analysis (e.g., create a region containing multiple people for crowd analysis, merge detections of objects in a scene into one region, combine multiple detections into a single monitoring zone), enabling multi-object region workflows
- **Nearby Detection Grouping**: Group nearby detections together into a single merged detection (e.g., merge detections of objects close to each other, group nearby detections into clusters, combine adjacent detections for simplified processing), enabling spatial grouping workflows
- **Detection Simplification**: Simplify multiple detections into one larger detection for downstream processing (e.g., reduce multiple detections to one for simpler analysis, consolidate detections for easier visualization, merge detections for streamlined workflows), enabling detection simplification workflows
- **Zone Definition from Detections**: Create zone boundaries from multiple detection locations (e.g., define zones based on detection locations, create regions from detected object positions, establish boundaries from detection clusters), enabling zone creation from detections
- **Redundant Detection Removal**: Merge redundant or duplicate detections into a single representation (e.g., combine duplicate detections from different stages, merge redundant object detections, consolidate repeated detections), enabling redundant detection consolidation workflows

## Connecting to Other Blocks

This block receives multiple detection predictions and produces a single merged detection:

- **After detection blocks** (e.g., Object Detection, Instance Segmentation, Keypoint Detection) to merge multiple detections into one unified detection for simplified processing, enabling detection consolidation workflows
- **After filtering blocks** (e.g., Detections Filter) to merge filtered detections that meet specific criteria into a single detection (e.g., merge filtered detections by class, combine detections after filtering, consolidate filtered results), enabling filtered detection consolidation
- **Before crop blocks** to create a single crop region from multiple detections (e.g., crop a region containing multiple objects, extract area encompassing multiple detections, create unified crop region), enabling multi-detection region extraction
- **Before zone-based blocks** (e.g., Polygon Zone, Dynamic Zone) to define zones based on merged detection regions (e.g., create zones from merged detection areas, establish monitoring zones from merged detections, define regions from consolidated detections), enabling zone creation from merged detections
- **Before visualization blocks** to display simplified merged detections instead of multiple individual detections (e.g., visualize consolidated detection regions, display merged bounding boxes, show simplified detection representation), enabling simplified visualization outputs
- **Before analysis blocks** that benefit from simplified detection representation (e.g., analyze merged detection regions, process consolidated detections, work with simplified detection data), enabling simplified detection analysis workflows


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/detections_merge@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `class_name` | `str` | Class name to assign to the merged detection. The merged detection will use this class name in its data. Default is 'merged_detection' to indicate that this is a merged detection. You can customize this to represent a specific category or to indicate the purpose of the merged detection (e.g., 'crowd', 'group', 'region'). This class name will be stored in the detection's data dictionary.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Detections Merge` in version `v1`.

    - inputs: [`SAM 3`](sam3.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Time in Zone`](timein_zone.md), [`Dynamic Crop`](dynamic_crop.md), [`Detections Filter`](detections_filter.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Object Detection Model`](object_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Velocity`](velocity.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Path Deviation`](path_deviation.md), [`Template Matching`](template_matching.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`EasyOCR`](easy_ocr.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Track Class Lock`](track_class_lock.md), [`Gaze Detection`](gaze_detection.md), [`Dynamic Zone`](dynamic_zone.md), [`YOLO-World Model`](yolo_world_model.md), [`Detections Combine`](detections_combine.md), [`Byte Tracker`](byte_tracker.md), [`Detections Transformation`](detections_transformation.md), [`Time in Zone`](timein_zone.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`OCR Model`](ocr_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Byte Tracker`](byte_tracker.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`VLM As Detector`](vlm_as_detector.md), [`SORT Tracker`](sort_tracker.md), [`PP-OCR`](ppocr.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Motion Detection`](motion_detection.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections Consensus`](detections_consensus.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`Detection Event Log`](detection_event_log.md), [`Detections Stitch`](detections_stitch.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Object Detection Model`](object_detection_model.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`SAM 3`](sam3.md), [`SAM 3`](sam3.md), [`Detections Merge`](detections_merge.md), [`Path Deviation`](path_deviation.md), [`Google Vision OCR`](google_vision_ocr.md), [`Perspective Correction`](perspective_correction.md), [`Moondream2`](moondream2.md), [`Detection Offset`](detection_offset.md), [`VLM As Detector`](vlm_as_detector.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Line Counter`](line_counter.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Overlap Filter`](overlap_filter.md), [`Seg Preview`](seg_preview.md), [`Byte Tracker`](byte_tracker.md)
    - outputs: [`Trace Visualization`](trace_visualization.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Time in Zone`](timein_zone.md), [`Dynamic Crop`](dynamic_crop.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Detections Filter`](detections_filter.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Velocity`](velocity.md), [`Path Deviation`](path_deviation.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Frame Delay`](frame_delay.md), [`Distance Measurement`](distance_measurement.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Track Class Lock`](track_class_lock.md), [`Florence-2 Model`](florence2_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Detections Combine`](detections_combine.md), [`Detections Transformation`](detections_transformation.md), [`Byte Tracker`](byte_tracker.md), [`Time in Zone`](timein_zone.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Circle Visualization`](circle_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Byte Tracker`](byte_tracker.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`SORT Tracker`](sort_tracker.md), [`Camera Focus`](camera_focus.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Size Measurement`](size_measurement.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Color Visualization`](color_visualization.md), [`Detections Consensus`](detections_consensus.md), [`Overlap Analysis`](overlap_analysis.md), [`Triangle Visualization`](triangle_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Detections Stitch`](detections_stitch.md), [`Background Color Visualization`](background_color_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Time in Zone`](timein_zone.md), [`Blur Visualization`](blur_visualization.md), [`Detections Merge`](detections_merge.md), [`Florence-2 Model`](florence2_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Path Deviation`](path_deviation.md), [`GeoTag Detection`](geo_tag_detection.md), [`Perspective Correction`](perspective_correction.md), [`Label Visualization`](label_visualization.md), [`Detection Offset`](detection_offset.md), [`Line Counter`](line_counter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Overlap Filter`](overlap_filter.md), [`Event Writer`](event_writer.md), [`Line Counter`](line_counter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Detections Merge` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `predictions` (*Union[[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md)]*): Detection predictions containing multiple detections to merge into a single detection. Supports object detection, instance segmentation, or keypoint detection predictions. All input detections will be combined into one merged detection with a union bounding box that encompasses all input detections. If empty detections are provided, the block returns an empty detection result. The merged detection will contain all input detections within its bounding box boundaries..

    - output
    
        - `predictions` ([`object_detection_prediction`](../kinds/object_detection_prediction.md)): Prediction with detected bounding boxes in form of sv.Detections(...) object.



??? tip "Example JSON definition of step `Detections Merge` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/detections_merge@v1",
	    "predictions": "$steps.object_detection_model.predictions",
	    "class_name": "<block_does_not_provide_example>"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    


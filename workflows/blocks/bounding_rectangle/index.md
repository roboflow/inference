
# Bounding Rectangle



??? "Class: `BoundingRectBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/transformations/bounding_rect/v1.py">inference.core.workflows.core_steps.transformations.bounding_rect.v1.BoundingRectBlockV1</a>
    



Calculate the minimum rotated bounding rectangle around polygon segmentation masks, converting complex polygon shapes into simplified rectangular bounding boxes with orientation information for zone creation, region simplification, and rectangular area approximation based on detected object shapes.

## How This Block Works

This block processes instance segmentation predictions with polygon masks and calculates the minimum rotated bounding rectangle (smallest rectangle that can enclose the polygon) for each detection. The block:

1. Receives instance segmentation predictions containing polygon masks (validates that masks are present)
2. Processes each detection's polygon mask individually
3. Extracts the largest contour from the polygon mask (handles multi-contour masks by selecting the largest)
4. Calculates the minimum rotated bounding rectangle using OpenCV's minAreaRect:
   - Finds the smallest rotated rectangle that can completely enclose the polygon
   - Determines the rectangle's center point, dimensions (width, height), and rotation angle
   - Computes the four corner points of the rotated rectangle
5. Updates the detection's mask to be a rectangular mask matching the calculated bounding rectangle (converts the rotated rectangle polygon back to a mask format)
6. Updates the detection's axis-aligned bounding box (xyxy) to the bounding box of the rotated rectangle (fits the rotated rectangle into an axis-aligned box)
7. Stores additional rectangle metadata in the detection data:
   - Rectangle corner coordinates (rotated rectangle points)
   - Rectangle width and height (dimensions of the rotated rectangle)
   - Rectangle angle (rotation angle in degrees)
8. Merges all processed detections and returns them with updated masks, bounding boxes, and rectangle metadata

The block transforms complex polygon shapes into simplified rectangular representations, preserving orientation information through the rotation angle. This is particularly useful when you need to create zones or regions based on detected object shapes (e.g., sports fields, road segments, marked areas) and want to simplify them to rectangular approximations. The minimum rotated rectangle provides the most compact rectangular representation of the polygon, potentially at an angle to minimize area.

## Common Use Cases

- **Zone Creation from Object Shapes**: Convert detected polygon shapes into rectangular zones for area monitoring or analysis (e.g., create zones from basketball court detections, generate road segment zones from road markings, create rectangular regions from zebra crossing detections), enabling zone-based workflows from complex shapes
- **Region Simplification**: Simplify complex polygon shapes to rectangular approximations for easier processing (e.g., simplify irregular segmentation masks to rectangles, convert complex shapes to rectangular regions, approximate polygon areas with rectangles), enabling simplified region processing
- **Rotated Region Detection**: Detect and extract rotated rectangular regions from polygon detections (e.g., find rotated parking spaces from segmentation, detect angled road markings as rectangles, extract rotated objects as rectangular zones), enabling rotation-aware region extraction
- **Area Approximation**: Approximate polygon areas with compact rectangular bounding boxes (e.g., approximate sports field areas with minimal rectangles, estimate region sizes using rotated bounding boxes, calculate compact rectangular areas from complex shapes), enabling area estimation with rectangular approximations
- **Shape Normalization**: Normalize polygon shapes to rectangular representations for standardized processing (e.g., normalize detected shapes to rectangles for consistent analysis, standardize polygon regions to rectangular format, convert variable shapes to uniform rectangular regions), enabling shape normalization workflows
- **Multi-Object Zone Extraction**: Extract rectangular zones from multiple detected polygon objects (e.g., create zones from multiple road segment detections, generate rectangular regions from multiple field detections, extract zones from various marked area detections), enabling multi-object zone creation workflows

## Connecting to Other Blocks

This block receives instance segmentation predictions with polygon masks and produces detections with rectangular masks and bounding boxes:

- **After instance segmentation blocks** to convert polygon masks to rectangular bounding boxes for zone creation or simplified processing, enabling rectangular zone generation from complex shapes
- **Before zone-based blocks** (e.g., Polygon Zone, Dynamic Zone) to prepare rectangular regions for zone-based workflows (e.g., create zones from simplified rectangles, use rectangular approximations for zone monitoring, enable zone workflows with rectangular regions), enabling zone-based workflows with simplified shapes
- **After filtering blocks** (e.g., Detections Filter) to process only specific polygon detections before converting to rectangles (e.g., filter detections by class before rectangular conversion, select specific polygon types for rectangle extraction, prepare filtered detections for zone creation), enabling selective rectangle extraction
- **Before crop blocks** to extract rectangular regions from polygon detections (e.g., crop rotated rectangular regions from polygon shapes, extract rectangular areas from complex detections, prepare rectangular crop regions from polygons), enabling rectangular region extraction
- **Before visualization blocks** to display simplified rectangular representations of complex polygons (e.g., visualize rectangular approximations of polygons, display rotated bounding rectangles, show simplified rectangular zones), enabling rectangular visualization outputs
- **Before analysis blocks** that work better with rectangular regions than complex polygons (e.g., analyze rectangular zones instead of polygons, process simplified rectangular regions, work with normalized rectangular shapes), enabling simplified region analysis workflows


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/bounding_rect@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Bounding Rectangle` in version `v1`.

    - inputs: [`Detections Stitch`](detections_stitch.md), [`Seg Preview`](seg_preview.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Detections Filter`](detections_filter.md), [`Dynamic Zone`](dynamic_zone.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections Combine`](detections_combine.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Detection Event Log`](detection_event_log.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Detections Consensus`](detections_consensus.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Time in Zone`](timein_zone.md), [`Perspective Correction`](perspective_correction.md), [`Velocity`](velocity.md), [`Detection Offset`](detection_offset.md), [`SAM 3`](sam3.md), [`SAM 3`](sam3.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Dynamic Crop`](dynamic_crop.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`SORT Tracker`](sort_tracker.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Detections Transformation`](detections_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Path Deviation`](path_deviation.md), [`Time in Zone`](timein_zone.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Path Deviation`](path_deviation.md), [`SAM 3`](sam3.md)
    - outputs: [`Detections Stabilizer`](detections_stabilizer.md), [`Detections Stitch`](detections_stitch.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Distance Measurement`](distance_measurement.md), [`Color Visualization`](color_visualization.md), [`Detections Combine`](detections_combine.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Byte Tracker`](byte_tracker.md), [`Time in Zone`](timein_zone.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Detections Consensus`](detections_consensus.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Camera Focus`](camera_focus.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Detection Offset`](detection_offset.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Size Measurement`](size_measurement.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Florence-2 Model`](florence2_model.md), [`Halo Visualization`](halo_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Crop Visualization`](crop_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Path Deviation`](path_deviation.md), [`Time in Zone`](timein_zone.md), [`Dot Visualization`](dot_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Path Deviation`](path_deviation.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Detections Filter`](detections_filter.md), [`Icon Visualization`](icon_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Dynamic Zone`](dynamic_zone.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Blur Visualization`](blur_visualization.md), [`Detections Merge`](detections_merge.md), [`Perspective Correction`](perspective_correction.md), [`Overlap Filter`](overlap_filter.md), [`Line Counter`](line_counter.md), [`Velocity`](velocity.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Label Visualization`](label_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Triangle Visualization`](triangle_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Bounding Rectangle` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `predictions` (*[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)*): Instance segmentation predictions containing polygon masks. The block requires masks to be present - it will raise an error if masks are missing. Each detection's polygon mask will be processed to calculate the minimum rotated bounding rectangle. The mask should contain polygon shapes that you want to convert to rectangular bounding boxes. Detections are processed individually, with the largest contour extracted from each mask. The block outputs detections with updated masks (rectangular), updated bounding boxes (axis-aligned boxes of the rotated rectangles), and additional rectangle metadata stored in detection.data (rectangle coordinates, width, height, angle)..

    - output
    
        - `detections_with_rect` ([`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)): Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object.



??? tip "Example JSON definition of step `Bounding Rectangle` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/bounding_rect@v1",
	    "predictions": "$segmentation.predictions"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    


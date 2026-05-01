
# Detection Offset



??? "Class: `DetectionOffsetBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/transformations/detection_offset/v1.py">inference.core.workflows.core_steps.transformations.detection_offset.v1.DetectionOffsetBlockV1</a>
    



Expand or contract detection bounding boxes by applying fixed offsets to their width and height, adding padding around detections to include more context, adjust bounding box sizes for downstream processing, or compensate for tight detections, supporting both pixel-based and percentage-based offset units for flexible bounding box adjustment.

## How This Block Works

This block adjusts the size of detection bounding boxes by adding offsets to their dimensions, effectively expanding or contracting the boxes to include more or less context around detected objects. The block:

1. Receives detection predictions (object detection, instance segmentation, or keypoint detection) containing bounding boxes
2. Processes each detection's bounding box coordinates independently
3. Calculates offsets based on the selected unit type:
   - **Pixel-based offsets**: Adds/subtracts a fixed number of pixels on each side (offset_width//2 pixels on left/right, offset_height//2 pixels on top/bottom)
   - **Percentage-based offsets**: Calculates offsets as a percentage of the bounding box's width and height (offset_width% of box width, offset_height% of box height)
4. Applies the offsets to expand the bounding boxes:
   - Subtracts half the width offset from x_min and adds half to x_max (expands horizontally)
   - Subtracts half the height offset from y_min and adds half to y_max (expands vertically)
5. Clips the adjusted bounding boxes to image boundaries (ensures coordinates stay within image dimensions using min/max constraints)
6. Updates detection metadata:
   - Sets parent_id_key to reference the original detection IDs (preserves traceability)
   - Generates new detection IDs for the offset detections (tracks that these are modified versions)
7. Preserves all other detection properties (masks, keypoints, polygons, class labels, confidence scores) unchanged
8. Returns the modified detections with expanded or contracted bounding boxes

The block applies offsets symmetrically around the center of each bounding box, expanding the box equally on all sides based on the width and height offsets. Positive offsets expand boxes (add padding), while the implementation always expands boxes outward. The pixel-based mode applies fixed pixel offsets regardless of box size, useful for consistent padding. The percentage-based mode applies offsets proportional to box size, useful when padding should scale with the detected object size. Boxes are automatically clipped to image boundaries to prevent invalid coordinates.

## Common Use Cases

- **Context Padding for Analysis**: Expand tight bounding boxes to include more surrounding context (e.g., add padding around detected objects for better classification, expand boxes to include object context for feature extraction, add margin around text detections for OCR), enabling improved analysis with additional context
- **Detection Size Adjustment**: Adjust bounding box sizes to match downstream processing requirements (e.g., expand boxes for models that need larger input regions, adjust box sizes to accommodate specific analysis needs, modify detections for compatibility with other blocks), enabling size customization for workflow compatibility
- **Tight Detection Compensation**: Expand overly tight bounding boxes that cut off parts of objects (e.g., add padding to tight object detections, expand boxes that miss object edges, compensate for models that produce undersized boxes), enabling better object coverage
- **Multi-Stage Workflow Preparation**: Prepare detections with adjusted sizes for secondary processing (e.g., expand initial detections before running secondary models, adjust box sizes for specialized analysis blocks, prepare detections with context for detailed processing), enabling optimized multi-stage workflows
- **Crop Region Optimization**: Adjust bounding boxes before cropping to include desired context (e.g., add padding before dynamic cropping to include surrounding context, expand boxes to capture more area for analysis, adjust crop regions for better feature extraction), enabling optimized region extraction
- **Visualization and Display**: Adjust bounding box sizes for better visualization or display purposes (e.g., expand boxes for clearer annotations, adjust box sizes for presentation, modify detections for visualization consistency), enabling improved visual outputs

## Connecting to Other Blocks

This block receives detection predictions and produces adjusted detections with modified bounding boxes:

- **After detection blocks** (e.g., Object Detection, Instance Segmentation, Keypoint Detection) to expand or adjust bounding box sizes before further processing, enabling size-optimized detections for downstream analysis
- **Before dynamic crop blocks** to adjust bounding box sizes before cropping, enabling optimized crop regions with desired context or padding
- **Before classification or analysis blocks** that benefit from additional context around detections (e.g., classification with context, feature extraction from expanded regions, detailed analysis with padding), enabling improved analysis with context
- **In multi-stage detection workflows** where initial detections need size adjustments before secondary processing (e.g., expand initial detections before running specialized models, adjust box sizes for compatibility, prepare detections for optimized processing), enabling flexible multi-stage workflows
- **Before visualization blocks** to adjust bounding box sizes for display purposes (e.g., expand boxes for clearer annotations, adjust sizes for presentation, modify detections for visualization consistency), enabling optimized visual outputs
- **Before blocks that process detection regions** where bounding box size matters (e.g., OCR on text regions with padding, feature extraction from expanded regions, specialized models requiring specific box sizes), enabling size-optimized region processing


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/detection_offset@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `offset_width` | `int` | Offset value to apply to bounding box width. Must be a positive integer. If units is 'Pixels', this is the number of pixels added to the box width (divided equally between left and right sides - offset_width//2 pixels on each side). If units is 'Percent (%)', this is the percentage of the bounding box width to add (calculated as percentage of the box's width, then divided between left and right). Positive values expand boxes horizontally. Boxes are clipped to image boundaries automatically.. | ✅ |
| `offset_height` | `int` | Offset value to apply to bounding box height. Must be a positive integer. If units is 'Pixels', this is the number of pixels added to the box height (divided equally between top and bottom sides - offset_height//2 pixels on each side). If units is 'Percent (%)', this is the percentage of the bounding box height to add (calculated as percentage of the box's height, then divided between top and bottom). Positive values expand boxes vertically. Boxes are clipped to image boundaries automatically.. | ✅ |
| `units` | `str` | Unit type for offset values: 'Pixels' for fixed pixel offsets (same number of pixels for all boxes regardless of size) or 'Percent (%)' for percentage-based offsets (proportional to each bounding box's dimensions). Pixel offsets provide consistent padding in absolute terms. Percentage offsets scale with box size, providing proportional padding. Use pixels when you need consistent absolute padding. Use percentage when padding should scale with detected object size.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Detection Offset` in version `v1`.

    - inputs: [`Detections Stitch`](detections_stitch.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Seg Preview`](seg_preview.md), [`Detections Filter`](detections_filter.md), [`Object Detection Model`](object_detection_model.md), [`Dynamic Zone`](dynamic_zone.md), [`Mask Edge Snap`](mask_edge_snap.md), [`OCR Model`](ocr_model.md), [`Gaze Detection`](gaze_detection.md), [`Image Contours`](image_contours.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Google Vision OCR`](google_vision_ocr.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Distance Measurement`](distance_measurement.md), [`EasyOCR`](easy_ocr.md), [`Detections Combine`](detections_combine.md), [`Object Detection Model`](object_detection_model.md), [`Motion Detection`](motion_detection.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Byte Tracker`](byte_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Bounding Rectangle`](bounding_rectangle.md), [`VLM As Detector`](vlm_as_detector.md), [`Byte Tracker`](byte_tracker.md), [`Detections Consensus`](detections_consensus.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Time in Zone`](timein_zone.md), [`Detections Merge`](detections_merge.md), [`Perspective Correction`](perspective_correction.md), [`Overlap Filter`](overlap_filter.md), [`Line Counter`](line_counter.md), [`Velocity`](velocity.md), [`Pixel Color Count`](pixel_color_count.md), [`Object Detection Model`](object_detection_model.md), [`Byte Tracker`](byte_tracker.md), [`YOLO-World Model`](yolo_world_model.md), [`Detection Offset`](detection_offset.md), [`SAM 3`](sam3.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`SAM 3`](sam3.md), [`VLM As Detector`](vlm_as_detector.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Template Matching`](template_matching.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Dynamic Crop`](dynamic_crop.md), [`SORT Tracker`](sort_tracker.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Moondream2`](moondream2.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SIFT Comparison`](sift_comparison.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Detections Transformation`](detections_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Path Deviation`](path_deviation.md), [`Time in Zone`](timein_zone.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Path Deviation`](path_deviation.md), [`SAM 3`](sam3.md), [`SIFT Comparison`](sift_comparison.md)
    - outputs: [`Detections Stabilizer`](detections_stabilizer.md), [`Detections Stitch`](detections_stitch.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Distance Measurement`](distance_measurement.md), [`Color Visualization`](color_visualization.md), [`Detections Combine`](detections_combine.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Ellipse Visualization`](ellipse_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Polygon Visualization`](polygon_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Byte Tracker`](byte_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Detections Consensus`](detections_consensus.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Time in Zone`](timein_zone.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Trace Visualization`](trace_visualization.md), [`Camera Focus`](camera_focus.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Detection Offset`](detection_offset.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Size Measurement`](size_measurement.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Florence-2 Model`](florence2_model.md), [`Halo Visualization`](halo_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Crop Visualization`](crop_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Path Deviation`](path_deviation.md), [`Time in Zone`](timein_zone.md), [`Dot Visualization`](dot_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Path Deviation`](path_deviation.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Icon Visualization`](icon_visualization.md), [`Detections Filter`](detections_filter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Dynamic Zone`](dynamic_zone.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Blur Visualization`](blur_visualization.md), [`Detections Merge`](detections_merge.md), [`Perspective Correction`](perspective_correction.md), [`Overlap Filter`](overlap_filter.md), [`Line Counter`](line_counter.md), [`Velocity`](velocity.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Label Visualization`](label_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Stitch OCR Detections`](stitch_ocr_detections.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Detection Offset` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `predictions` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Detection predictions containing bounding boxes to adjust. Supports object detection, instance segmentation, or keypoint detection predictions. The bounding boxes in these predictions will be expanded or contracted based on the offset_width and offset_height values. All detection properties (masks, keypoints, polygons, classes, confidence) are preserved unchanged - only bounding box coordinates are modified..
        - `offset_width` (*[`integer`](../kinds/integer.md)*): Offset value to apply to bounding box width. Must be a positive integer. If units is 'Pixels', this is the number of pixels added to the box width (divided equally between left and right sides - offset_width//2 pixels on each side). If units is 'Percent (%)', this is the percentage of the bounding box width to add (calculated as percentage of the box's width, then divided between left and right). Positive values expand boxes horizontally. Boxes are clipped to image boundaries automatically..
        - `offset_height` (*[`integer`](../kinds/integer.md)*): Offset value to apply to bounding box height. Must be a positive integer. If units is 'Pixels', this is the number of pixels added to the box height (divided equally between top and bottom sides - offset_height//2 pixels on each side). If units is 'Percent (%)', this is the percentage of the bounding box height to add (calculated as percentage of the box's height, then divided between top and bottom). Positive values expand boxes vertically. Boxes are clipped to image boundaries automatically..

    - output
    
        - `predictions` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md)]*): Prediction with detected bounding boxes in form of sv.Detections(...) object if `object_detection_prediction` or Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object if `instance_segmentation_prediction` or Prediction with detected bounding boxes and detected keypoints in form of sv.Detections(...) object if `keypoint_detection_prediction`.



??? tip "Example JSON definition of step `Detection Offset` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/detection_offset@v1",
	    "predictions": "$steps.object_detection_model.predictions",
	    "offset_width": 10,
	    "offset_height": 10,
	    "units": "Pixels"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    


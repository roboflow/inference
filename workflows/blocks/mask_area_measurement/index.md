
# Mask Area Measurement



??? "Class: `MaskAreaMeasurementBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/classical_cv/mask_area_measurement/v1.py">inference.core.workflows.core_steps.classical_cv.mask_area_measurement.v1.MaskAreaMeasurementBlockV1</a>
    



Measure the area of detected objects. For instance segmentation masks, the area is computed by counting non-zero mask pixels (correctly handling holes). For bounding-box-only detections, the area is width multiplied by height. Optionally converts pixel areas to real-world units using a `pixels_per_unit` calibration value.

## How This Block Works

This block calculates the area of each detected object and stores two values per detection:

- **`area_px`** — area in square pixels (always computed)
- **`area_converted`** — area in real-world units: `area_px / (pixels_per_unit ** 2)` (equals `area_px` when `pixels_per_unit` is 1.0)

Both values are attached to each detection and included in the serialized JSON output. The block returns the input detections with these fields added, so downstream blocks (e.g., label visualization) can display the area values.

### Area Computation

The block operates in two modes depending on the type of predictions it receives:

1. **Mask Pixel Area (Instance Segmentation)**: When the input detections include segmentation masks, the block counts the non-zero pixels in each mask using `cv2.countNonZero`. This correctly handles masks with holes — hole pixels are zero and are excluded from the count.

2. **Bounding Box Area (Object Detection)**: When no segmentation mask is available, the block falls back to computing the area as the bounding box width multiplied by height (`w * h`).

### Unit Conversion

Set the `pixels_per_unit` input to convert pixel areas to real-world units (e.g., cm², in², mm²). Because area is two-dimensional, the conversion squares the ratio:

```
area_converted = area_px / (pixels_per_unit ** 2)
```

For example, if your calibration is 130 pixels/cm, a detection with `area_px = 16900` would have `area_converted = 16900 / 16900 = 1.0 cm²`.

**How to determine pixels_per_unit:** Place an object of known size in the camera's field of view (e.g., a ruler or calibration target). Measure its length in pixels in the image and divide by its real-world length. For instance, if a 10 cm reference object spans 1300 pixels, then `pixels_per_cm = 1300 / 10 = 130`. If you are using perspective correction, the calibration object must be placed on the same plane from which the perspective correction was calculated.

## Common Use Cases

- **Size-Based Filtering**: Filter out small noise detections by chaining with a filtering block to keep only detections above a minimum area threshold.
- **Quality Control**: Verify that manufactured components meet size specifications by comparing measured areas against expected ranges.
- **Agricultural Analysis**: Measure leaf area, crop coverage, or canopy extent from aerial or close-up imagery.
- **Medical Imaging**: Quantify the area of wounds, lesions, or anatomical structures. Use `pixels_per_unit` to get real-world measurements for clinical documentation.

## Connecting to Other Blocks

- **Upstream -- Detection and Segmentation Models**: Connect the output of an object detection or instance segmentation model to the `predictions` input. Instance segmentation models (which produce masks) yield more accurate area measurements than bounding-box-only detections.
- **Upstream -- Camera Calibration Block**: Use `roboflow_core/camera_calibration@v1` upstream to correct lens distortion before detection.
- **Upstream -- Perspective Correction Block**: Use `roboflow_core/perspective_correction@v1` upstream to transform angled images to a top-down view so that area measurements reflect true object footprints.
- **Downstream -- Visualization**: Pass the output `predictions` to label or polygon visualization blocks. The `area_px` and `area_converted` fields are available for display as labels.
- **Downstream -- Filtering Blocks**: Use the enriched detections with a filtering block to keep only detections whose area meets a threshold.

## Requirements

This block requires detection predictions from an object detection or instance segmentation model. No additional environment variables, API keys, or external dependencies are needed beyond OpenCV and NumPy (included with inference). For the most accurate area measurements, use instance segmentation models that produce per-object masks.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/mask_area_measurement@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `pixels_per_unit` | `float` | Number of pixels per real-world unit of length (e.g., pixels per cm). The converted area is computed as area_px / (pixels_per_unit ** 2). Default 1.0 means no conversion (area_converted equals area_px).. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Mask Area Measurement` in version `v1`.

    - inputs: [`Detections Stitch`](detections_stitch.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Seg Preview`](seg_preview.md), [`Detections Filter`](detections_filter.md), [`Object Detection Model`](object_detection_model.md), [`Dynamic Zone`](dynamic_zone.md), [`Mask Edge Snap`](mask_edge_snap.md), [`OCR Model`](ocr_model.md), [`Gaze Detection`](gaze_detection.md), [`Google Vision OCR`](google_vision_ocr.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`EasyOCR`](easy_ocr.md), [`Detections Combine`](detections_combine.md), [`Object Detection Model`](object_detection_model.md), [`Motion Detection`](motion_detection.md), [`Cosine Similarity`](cosine_similarity.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Detection Event Log`](detection_event_log.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Byte Tracker`](byte_tracker.md), [`VLM As Detector`](vlm_as_detector.md), [`Byte Tracker`](byte_tracker.md), [`Detections Consensus`](detections_consensus.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Time in Zone`](timein_zone.md), [`Detections Merge`](detections_merge.md), [`Perspective Correction`](perspective_correction.md), [`Overlap Filter`](overlap_filter.md), [`Velocity`](velocity.md), [`Object Detection Model`](object_detection_model.md), [`Identify Changes`](identify_changes.md), [`Camera Focus`](camera_focus.md), [`Byte Tracker`](byte_tracker.md), [`YOLO-World Model`](yolo_world_model.md), [`Detection Offset`](detection_offset.md), [`SAM 3`](sam3.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`SAM 3`](sam3.md), [`VLM As Detector`](vlm_as_detector.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Template Matching`](template_matching.md), [`Dynamic Crop`](dynamic_crop.md), [`SORT Tracker`](sort_tracker.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Moondream2`](moondream2.md), [`Camera Focus`](camera_focus.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Detections Transformation`](detections_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Path Deviation`](path_deviation.md), [`Time in Zone`](timein_zone.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Path Deviation`](path_deviation.md), [`SAM 3`](sam3.md)
    - outputs: [`Detections Stabilizer`](detections_stabilizer.md), [`Detections Stitch`](detections_stitch.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Distance Measurement`](distance_measurement.md), [`Color Visualization`](color_visualization.md), [`Detections Combine`](detections_combine.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Ellipse Visualization`](ellipse_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Polygon Visualization`](polygon_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Byte Tracker`](byte_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Detections Consensus`](detections_consensus.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Time in Zone`](timein_zone.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Trace Visualization`](trace_visualization.md), [`Camera Focus`](camera_focus.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Detection Offset`](detection_offset.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Size Measurement`](size_measurement.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Florence-2 Model`](florence2_model.md), [`Halo Visualization`](halo_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Crop Visualization`](crop_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Path Deviation`](path_deviation.md), [`Time in Zone`](timein_zone.md), [`Dot Visualization`](dot_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Path Deviation`](path_deviation.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Icon Visualization`](icon_visualization.md), [`Detections Filter`](detections_filter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Dynamic Zone`](dynamic_zone.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Blur Visualization`](blur_visualization.md), [`Detections Merge`](detections_merge.md), [`Perspective Correction`](perspective_correction.md), [`Overlap Filter`](overlap_filter.md), [`Line Counter`](line_counter.md), [`Velocity`](velocity.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Label Visualization`](label_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Triangle Visualization`](triangle_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Stitch OCR Detections`](stitch_ocr_detections.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Mask Area Measurement` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `predictions` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Detection predictions to measure areas for..
        - `pixels_per_unit` (*[`float`](../kinds/float.md)*): Number of pixels per real-world unit of length (e.g., pixels per cm). The converted area is computed as area_px / (pixels_per_unit ** 2). Default 1.0 means no conversion (area_converted equals area_px)..

    - output
    
        - `predictions` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Prediction with detected bounding boxes in form of sv.Detections(...) object if `object_detection_prediction` or Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object if `instance_segmentation_prediction`.



??? tip "Example JSON definition of step `Mask Area Measurement` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/mask_area_measurement@v1",
	    "predictions": "$steps.model.predictions",
	    "pixels_per_unit": 1.0
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    


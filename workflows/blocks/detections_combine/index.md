
# Detections Combine



??? "Class: `DetectionsCombineBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/transformations/detections_combine/v1.py">inference.core.workflows.core_steps.transformations.detections_combine.v1.DetectionsCombineBlockV1</a>
    



Combine two sets of detection predictions into a single unified set of detections by merging both detection sets together, preserving all detections from both inputs for multi-source detection aggregation, combining results from multiple models, and consolidating detection sets from different processing stages into one workflow output.

## How This Block Works

This block combines two separate sets of detection predictions into a single unified detection set by merging all detections from both inputs. The block:

1. Receives two separate detection prediction sets (prediction_one and prediction_two), each containing multiple detections from object detection or instance segmentation models
2. Processes both detection sets independently (each set maintains its own detections, properties, masks, and metadata)
3. Merges the two detection sets using supervision's Detections.merge() method:
   - Combines all detections from prediction_one with all detections from prediction_two
   - Preserves all detection properties from both sets (bounding boxes, masks, classes, confidence scores, metadata)
   - Maintains detection order (typically prediction_one detections followed by prediction_two detections)
   - Handles all detection attributes including masks (for instance segmentation), keypoints, class IDs, class names, confidence scores, and custom data fields
4. Returns a single unified detection set containing all detections from both inputs

The block simply concatenates the two detection sets together, preserving all detections and their properties from both sources. Unlike the Detections Merge block (which creates a union bounding box from multiple detections), this block maintains all individual detections from both sets in the output. This is useful for combining detections from different models, different processing stages, or different detection sources into a single workflow stream for unified downstream processing.

## Common Use Cases

- **Multi-Model Detection Aggregation**: Combine detections from multiple detection models into a single unified set (e.g., combine detections from different object detection models, merge results from specialized models, aggregate detections from multiple model outputs), enabling multi-model detection workflows
- **Multi-Stage Detection Combination**: Combine detections from different processing stages or workflow branches (e.g., merge detections from different workflow paths, combine initial detections with refined detections, aggregate detections from multiple processing stages), enabling multi-stage detection aggregation
- **Detection Source Consolidation**: Consolidate detections from different sources or inputs into one set (e.g., combine detections from multiple images or frames, merge detections from different regions, aggregate detections from various sources), enabling detection source unification
- **Classification and Detection Combination**: Combine object detection results with classification results or other detection types (e.g., merge object detections with classification outputs, combine different detection types, aggregate complementary detection sets), enabling multi-type detection workflows
- **Filtered and Unfiltered Detection Combination**: Combine filtered detections with unfiltered detections or combine different filtered subsets (e.g., merge filtered detections by different criteria, combine specific class detections with general detections, aggregate different filtered detection sets), enabling flexible detection combination workflows
- **Workflow Branch Merging**: Merge detection results from different workflow branches back into a single detection stream (e.g., combine parallel processing branch results, merge conditional workflow paths, aggregate branch detection outputs), enabling workflow branch consolidation

## Connecting to Other Blocks

This block receives two detection prediction sets and produces a single combined detection set:

- **After multiple detection blocks** to combine detections from different models into one unified set (e.g., combine detections from multiple object detection models, merge results from different segmentation models, aggregate detections from various model outputs), enabling multi-model detection aggregation workflows
- **After filtering blocks** to combine filtered detection subsets (e.g., merge detections filtered by different criteria, combine class-specific filtered detections, aggregate various filtered detection sets), enabling filtered detection combination workflows
- **At workflow merge points** where different workflow branches need to be combined (e.g., merge parallel processing branch results, combine conditional path outputs, aggregate branch detection streams), enabling workflow branch merging workflows
- **Before downstream processing blocks** that need unified detection sets (e.g., process combined detections together, visualize unified detection sets, analyze aggregated detections), enabling unified detection processing workflows
- **Before crop blocks** to process combined detections together (e.g., crop regions from combined detection sets, extract areas from aggregated detections, process unified detection regions), enabling combined detection region extraction
- **Before visualization blocks** to display unified detection sets (e.g., visualize combined detections from multiple sources, display aggregated detection results, show merged detection outputs), enabling unified detection visualization workflows


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/detections_combine@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Detections Combine` in version `v1`.

    - inputs: [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Path Deviation`](path_deviation.md), [`Detection Offset`](detection_offset.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Detections Transformation`](detections_transformation.md), [`Perspective Correction`](perspective_correction.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Google Vision OCR`](google_vision_ocr.md), [`Detections Merge`](detections_merge.md), [`Dynamic Zone`](dynamic_zone.md), [`Detections Combine`](detections_combine.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Object Detection Model`](object_detection_model.md), [`Bounding Rectangle`](bounding_rectangle.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Velocity`](velocity.md), [`Byte Tracker`](byte_tracker.md), [`Dynamic Crop`](dynamic_crop.md), [`Detections Stitch`](detections_stitch.md), [`Path Deviation`](path_deviation.md), [`Template Matching`](template_matching.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Motion Detection`](motion_detection.md), [`OCR Model`](ocr_model.md), [`Detections Filter`](detections_filter.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`YOLO-World Model`](yolo_world_model.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Detections Consensus`](detections_consensus.md), [`Byte Tracker`](byte_tracker.md), [`SAM 3`](sam3.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Moondream2`](moondream2.md), [`VLM As Detector`](vlm_as_detector.md), [`EasyOCR`](easy_ocr.md), [`Overlap Filter`](overlap_filter.md), [`SAM 3`](sam3.md), [`Time in Zone`](timein_zone.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`SORT Tracker`](sort_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Track Class Lock`](track_class_lock.md), [`Object Detection Model`](object_detection_model.md), [`Time in Zone`](timein_zone.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`SAM 3`](sam3.md), [`Seg Preview`](seg_preview.md)
    - outputs: [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Trace Visualization`](trace_visualization.md), [`Path Deviation`](path_deviation.md), [`Distance Measurement`](distance_measurement.md), [`Detection Offset`](detection_offset.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Icon Visualization`](icon_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Color Visualization`](color_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Corner Visualization`](corner_visualization.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Detections Merge`](detections_merge.md), [`Halo Visualization`](halo_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Detections Combine`](detections_combine.md), [`Size Measurement`](size_measurement.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Halo Visualization`](halo_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Event Writer`](event_writer.md), [`Background Color Visualization`](background_color_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Mask Visualization`](mask_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Velocity`](velocity.md), [`Label Visualization`](label_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Dot Visualization`](dot_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Path Deviation`](path_deviation.md), [`Detections Stitch`](detections_stitch.md), [`Dynamic Crop`](dynamic_crop.md), [`Circle Visualization`](circle_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Camera Focus`](camera_focus.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Florence-2 Model`](florence2_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Detections Filter`](detections_filter.md), [`Overlap Analysis`](overlap_analysis.md), [`Blur Visualization`](blur_visualization.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Detections Consensus`](detections_consensus.md), [`Byte Tracker`](byte_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Line Counter`](line_counter.md), [`Triangle Visualization`](triangle_visualization.md), [`Overlap Filter`](overlap_filter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Time in Zone`](timein_zone.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`SORT Tracker`](sort_tracker.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Track Class Lock`](track_class_lock.md), [`Time in Zone`](timein_zone.md), [`Polygon Visualization`](polygon_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Model Comparison Visualization`](model_comparison_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Detections Combine` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `prediction_one` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): First set of detection predictions to combine. Supports object detection or instance segmentation predictions. All detections from this set will be included in the output. Detection properties (bounding boxes, masks, classes, confidence scores, metadata) are preserved as-is. This set is combined with prediction_two to create the unified output. Detections from this set typically appear first in the merged output..
        - `prediction_two` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Second set of detection predictions to combine. Supports object detection or instance segmentation predictions. All detections from this set will be included in the output. Detection properties (bounding boxes, masks, classes, confidence scores, metadata) are preserved as-is. This set is combined with prediction_one to create the unified output. Detections from this set are merged with detections from prediction_one to form a single combined detection set..

    - output
    
        - `predictions` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Prediction with detected bounding boxes in form of sv.Detections(...) object if `object_detection_prediction` or Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object if `instance_segmentation_prediction`.



??? tip "Example JSON definition of step `Detections Combine` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/detections_combine@v1",
	    "prediction_one": "$steps.my_object_detection_model.predictions",
	    "prediction_two": "$steps.my_object_detection_model.predictions"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    


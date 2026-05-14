
# SAM2 Video Tracker



??? "Class: `SegmentAnything2VideoBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/segment_anything2_video/v1.py">inference.core.workflows.core_steps.models.foundation.segment_anything2_video.v1.SegmentAnything2VideoBlockV1</a>
    



Run Segment Anything 2 on a live video stream frame by frame, keeping
per-video temporal memory so object identities are preserved across
frames.

Feed box detections from an upstream detector (e.g. a YOLO block) as
prompts.  The block multiplexes a single SAM2 camera predictor across
many video streams by keying state on `video_metadata.video_identifier`;
depending on `prompt_mode`, it either re-seeds the prompts periodically
or simply propagates existing tracks.

Intended for use with `InferencePipeline`, which delivers one frame at
a time and tags each frame with video metadata.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/segment_anything_2_video@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `model_id` | `str` | Streaming SAM2 model id resolved by `inference_models`.  The `sam2video` family ships four Hiera backbone sizes; `small` is the default trade-off between speed and quality.. | ✅ |
| `prompt_mode` | `str` | When to consume `boxes` as SAM2 prompts.  `first_frame` prompts once per session and then tracks; `every_n_frames` re-seeds every `prompt_interval` frames; `every_frame` re-seeds every frame.  On frames where re-seeding does not happen, `boxes` is ignored and the block simply propagates.. | ❌ |
| `prompt_interval` | `int` | For `prompt_mode=every_n_frames`: re-prompt every N frames.. | ✅ |
| `threshold` | `float` | Minimum confidence for emitted masks.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `SAM2 Video Tracker` in version `v1`.

    - inputs: [`Triangle Visualization`](triangle_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Time in Zone`](timein_zone.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Dynamic Zone`](dynamic_zone.md), [`SORT Tracker`](sort_tracker.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Stitch Images`](stitch_images.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Detections Combine`](detections_combine.md), [`Google Vision OCR`](google_vision_ocr.md), [`Object Detection Model`](object_detection_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Image Blur`](image_blur.md), [`Path Deviation`](path_deviation.md), [`Image Stack`](image_stack.md), [`Distance Measurement`](distance_measurement.md), [`Background Subtraction`](background_subtraction.md), [`Byte Tracker`](byte_tracker.md), [`Camera Focus`](camera_focus.md), [`Morphological Transformation`](morphological_transformation.md), [`Identify Changes`](identify_changes.md), [`QR Code Generator`](qr_code_generator.md), [`Cosine Similarity`](cosine_similarity.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Image Preprocessing`](image_preprocessing.md), [`Detection Event Log`](detection_event_log.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Line Counter`](line_counter.md), [`Crop Visualization`](crop_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Relative Static Crop`](relative_static_crop.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Perspective Correction`](perspective_correction.md), [`Halo Visualization`](halo_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Seg Preview`](seg_preview.md), [`Line Counter`](line_counter.md), [`Contrast Equalization`](contrast_equalization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Image Contours`](image_contours.md), [`Text Display`](text_display.md), [`Icon Visualization`](icon_visualization.md), [`SIFT`](sift.md), [`Circle Visualization`](circle_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Overlap Filter`](overlap_filter.md), [`Gaze Detection`](gaze_detection.md), [`Velocity`](velocity.md), [`Image Slicer`](image_slicer.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Camera Focus`](camera_focus.md), [`Time in Zone`](timein_zone.md), [`Image Slicer`](image_slicer.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Motion Detection`](motion_detection.md), [`Polygon Visualization`](polygon_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`SAM 3`](sam3.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`YOLO-World Model`](yolo_world_model.md), [`Detections Stitch`](detections_stitch.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Image Threshold`](image_threshold.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Grid Visualization`](grid_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Color Visualization`](color_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Corner Visualization`](corner_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Label Visualization`](label_visualization.md), [`Detections Consensus`](detections_consensus.md), [`Time in Zone`](timein_zone.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detections Merge`](detections_merge.md), [`VLM As Detector`](vlm_as_detector.md), [`Dot Visualization`](dot_visualization.md), [`Detection Offset`](detection_offset.md), [`Camera Calibration`](camera_calibration.md), [`Morphological Transformation`](morphological_transformation.md), [`SAM 3`](sam3.md), [`OCR Model`](ocr_model.md), [`EasyOCR`](easy_ocr.md), [`Trace Visualization`](trace_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Path Deviation`](path_deviation.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`SIFT Comparison`](sift_comparison.md), [`Moondream2`](moondream2.md), [`Depth Estimation`](depth_estimation.md), [`Byte Tracker`](byte_tracker.md), [`Template Matching`](template_matching.md), [`Detections Filter`](detections_filter.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Pixel Color Count`](pixel_color_count.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md)
    - outputs: [`Triangle Visualization`](triangle_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Time in Zone`](timein_zone.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Dynamic Zone`](dynamic_zone.md), [`Florence-2 Model`](florence2_model.md), [`SORT Tracker`](sort_tracker.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Detections Combine`](detections_combine.md), [`Path Deviation`](path_deviation.md), [`Distance Measurement`](distance_measurement.md), [`Byte Tracker`](byte_tracker.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Camera Focus`](camera_focus.md), [`Detection Event Log`](detection_event_log.md), [`Line Counter`](line_counter.md), [`Crop Visualization`](crop_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Perspective Correction`](perspective_correction.md), [`Halo Visualization`](halo_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Line Counter`](line_counter.md), [`Mask Edge Snap`](mask_edge_snap.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Icon Visualization`](icon_visualization.md), [`Overlap Filter`](overlap_filter.md), [`Circle Visualization`](circle_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Velocity`](velocity.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Florence-2 Model`](florence2_model.md), [`Time in Zone`](timein_zone.md), [`Polygon Visualization`](polygon_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Detections Stitch`](detections_stitch.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Dynamic Crop`](dynamic_crop.md), [`Color Visualization`](color_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Label Visualization`](label_visualization.md), [`Detections Consensus`](detections_consensus.md), [`Time in Zone`](timein_zone.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detections Merge`](detections_merge.md), [`Dot Visualization`](dot_visualization.md), [`Size Measurement`](size_measurement.md), [`Detection Offset`](detection_offset.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Trace Visualization`](trace_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Path Deviation`](path_deviation.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Byte Tracker`](byte_tracker.md), [`Detections Filter`](detections_filter.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Byte Tracker`](byte_tracker.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`SAM2 Video Tracker` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `boxes` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md)]*): Bounding boxes to use as SAM2 prompts.  Only read on frames where the block re-prompts (see `prompt_mode`)..
        - `model_id` (*[`roboflow_model_id`](../kinds/roboflow_model_id.md)*): Streaming SAM2 model id resolved by `inference_models`.  The `sam2video` family ships four Hiera backbone sizes; `small` is the default trade-off between speed and quality..
        - `prompt_interval` (*[`integer`](../kinds/integer.md)*): For `prompt_mode=every_n_frames`: re-prompt every N frames..
        - `threshold` (*[`float`](../kinds/float.md)*): Minimum confidence for emitted masks..

    - output
    
        - `predictions` ([`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)): Prediction with detected bounding boxes and segmentation masks in form of sv.Detections(...) object.



??? tip "Example JSON definition of step `SAM2 Video Tracker` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/segment_anything_2_video@v1",
	    "images": "$inputs.image",
	    "boxes": "$steps.object_detection_model.predictions",
	    "model_id": "sam2video/tiny",
	    "prompt_mode": "<block_does_not_provide_example>",
	    "prompt_interval": 30,
	    "threshold": 0.0
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    


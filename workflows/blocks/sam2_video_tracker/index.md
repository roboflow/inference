
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

    - inputs: [`SAM 3`](sam3.md), [`Camera Focus`](camera_focus.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Seg Preview`](seg_preview.md), [`Time in Zone`](timein_zone.md), [`Perspective Correction`](perspective_correction.md), [`Crop Visualization`](crop_visualization.md), [`Detections Combine`](detections_combine.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Identify Changes`](identify_changes.md), [`Google Vision OCR`](google_vision_ocr.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Label Visualization`](label_visualization.md), [`SAM 3`](sam3.md), [`Path Deviation`](path_deviation.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Corner Visualization`](corner_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Motion Detection`](motion_detection.md), [`Line Counter`](line_counter.md), [`Dynamic Crop`](dynamic_crop.md), [`EasyOCR`](easy_ocr.md), [`Background Subtraction`](background_subtraction.md), [`Byte Tracker`](byte_tracker.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Contours`](image_contours.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`VLM As Detector`](vlm_as_detector.md), [`Camera Focus`](camera_focus.md), [`Detection Offset`](detection_offset.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Overlap Filter`](overlap_filter.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`SAM 3`](sam3.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Template Matching`](template_matching.md), [`Triangle Visualization`](triangle_visualization.md), [`Detections Merge`](detections_merge.md), [`SIFT Comparison`](sift_comparison.md), [`SIFT`](sift.md), [`QR Code Generator`](qr_code_generator.md), [`YOLO-World Model`](yolo_world_model.md), [`Dot Visualization`](dot_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Detections Stitch`](detections_stitch.md), [`Relative Static Crop`](relative_static_crop.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Velocity`](velocity.md), [`Detection Event Log`](detection_event_log.md), [`Distance Measurement`](distance_measurement.md), [`Byte Tracker`](byte_tracker.md), [`SIFT Comparison`](sift_comparison.md), [`Trace Visualization`](trace_visualization.md), [`Object Detection Model`](object_detection_model.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Icon Visualization`](icon_visualization.md), [`Time in Zone`](timein_zone.md), [`Path Deviation`](path_deviation.md), [`Halo Visualization`](halo_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Color Visualization`](color_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Detections Filter`](detections_filter.md), [`Moondream2`](moondream2.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Line Counter`](line_counter.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Polygon Visualization`](polygon_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Dynamic Zone`](dynamic_zone.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Image Slicer`](image_slicer.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Morphological Transformation`](morphological_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Circle Visualization`](circle_visualization.md), [`Image Slicer`](image_slicer.md), [`Image Preprocessing`](image_preprocessing.md), [`Mask Visualization`](mask_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Grid Visualization`](grid_visualization.md), [`Image Threshold`](image_threshold.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Time in Zone`](timein_zone.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Text Display`](text_display.md), [`Camera Calibration`](camera_calibration.md), [`Cosine Similarity`](cosine_similarity.md), [`Object Detection Model`](object_detection_model.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Stitch Images`](stitch_images.md), [`Morphological Transformation`](morphological_transformation.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`VLM As Detector`](vlm_as_detector.md), [`Gaze Detection`](gaze_detection.md), [`OCR Model`](ocr_model.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Blur`](image_blur.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SORT Tracker`](sort_tracker.md), [`Depth Estimation`](depth_estimation.md), [`Detections Consensus`](detections_consensus.md)
    - outputs: [`Camera Focus`](camera_focus.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Time in Zone`](timein_zone.md), [`Perspective Correction`](perspective_correction.md), [`Crop Visualization`](crop_visualization.md), [`Detections Combine`](detections_combine.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Label Visualization`](label_visualization.md), [`Path Deviation`](path_deviation.md), [`Corner Visualization`](corner_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Byte Tracker`](byte_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Detection Offset`](detection_offset.md), [`Overlap Filter`](overlap_filter.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Triangle Visualization`](triangle_visualization.md), [`Detections Merge`](detections_merge.md), [`Dot Visualization`](dot_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Distance Measurement`](distance_measurement.md), [`Detection Event Log`](detection_event_log.md), [`Velocity`](velocity.md), [`Byte Tracker`](byte_tracker.md), [`Trace Visualization`](trace_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Size Measurement`](size_measurement.md), [`Icon Visualization`](icon_visualization.md), [`Time in Zone`](timein_zone.md), [`Path Deviation`](path_deviation.md), [`Halo Visualization`](halo_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Byte Tracker`](byte_tracker.md), [`Color Visualization`](color_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Detections Filter`](detections_filter.md), [`Florence-2 Model`](florence2_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Line Counter`](line_counter.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Polygon Visualization`](polygon_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Polygon Visualization`](polygon_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Circle Visualization`](circle_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Time in Zone`](timein_zone.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Line Counter`](line_counter.md), [`SORT Tracker`](sort_tracker.md), [`Detections Consensus`](detections_consensus.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`SAM2 Video Tracker` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `boxes` (*Union[[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md)]*): Bounding boxes to use as SAM2 prompts.  Only read on frames where the block re-prompts (see `prompt_mode`)..
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


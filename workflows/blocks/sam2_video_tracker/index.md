
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
| `model_id` | `str` | Streaming video tracker model id resolved by `inference_models`.  The `sam2video` family ships four Hiera backbone sizes; `small` is the default trade-off between speed and quality.  `sam3trackervideo` is SAM3's visually prompted tracker — the same prompt contract with a larger backbone, markedly better at identity retention on long videos and crowded scenes, at higher compute cost.. | ✅ |
| `prompt_mode` | `str` | When to consume `boxes` as SAM2 prompts.  `first_frame` prompts once per session and then tracks; `every_n_frames` re-seeds every `prompt_interval` frames; `every_frame` re-seeds every frame.  On frames where re-seeding does not happen, `boxes` is ignored and the block simply propagates.. | ❌ |
| `prompt_interval` | `int` | For `prompt_mode=every_n_frames`: re-prompt every N frames.. | ✅ |
| `threshold` | `float` | Minimum confidence for emitted masks.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-alert-circle-outline:{ style="color: #f57c00" } `soft` — runtime `hosted_serverless`, `dedicated_deployment`; execution `remote`; input `video`
:   Block keeps per-video state in process memory (keyed by video_metadata.video_identifier). With remote step execution on stateless or multi-replica HTTP runtimes, successive requests may be served by different worker processes, so the state resets between calls and the output is meaningless for tracking / counting / aggregation. Use local step execution in an InferencePipeline for stable cross-frame results.

:material-shield-alert:{ style="color: #d32f2f" } `hard` — runtime `self_hosted_cpu`; execution `local`
:   Requires a GPU; the streaming SAM2 video model needs CUDA.

:material-alert-circle-outline:{ style="color: #f57c00" } `soft` — input `image`
:   Block depends on temporal context from video or repeated-frame workflows. With a still image/photo, there is no meaningful history to track, compare, aggregate, or visualize, so the block provides little or no benefit.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `SAM2 Video Tracker` in version `v1`.

    - inputs: [`Image Blur`](image_blur.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Google Vision OCR`](google_vision_ocr.md), [`Byte Tracker`](byte_tracker.md), [`Image Threshold`](image_threshold.md), [`Morphological Transformation`](morphological_transformation.md), [`VLM As Detector`](vlm_as_detector.md), [`Triangle Visualization`](triangle_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Detections Combine`](detections_combine.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Detections Merge`](detections_merge.md), [`Cosine Similarity`](cosine_similarity.md), [`Grid Visualization`](grid_visualization.md), [`Mask Visualization`](mask_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Camera Focus`](camera_focus.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections Transformation`](detections_transformation.md), [`Template Matching`](template_matching.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Track Class Lock`](track_class_lock.md), [`Dynamic Zone`](dynamic_zone.md), [`SAM 3`](sam3.md), [`Dynamic Crop`](dynamic_crop.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Velocity`](velocity.md), [`SAM 3`](sam3.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`QR Code Generator`](qr_code_generator.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Color Visualization`](color_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`YOLO-World Model`](yolo_world_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Halo Visualization`](halo_visualization.md), [`Label Visualization`](label_visualization.md), [`SIFT`](sift.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Byte Tracker`](byte_tracker.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Detection Offset`](detection_offset.md), [`Object Detection Model`](object_detection_model.md), [`Path Deviation`](path_deviation.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Seg Preview`](seg_preview.md), [`Morphological Transformation`](morphological_transformation.md), [`Text Display`](text_display.md), [`Gaze Detection`](gaze_detection.md), [`Overlap Filter`](overlap_filter.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Detections Filter`](detections_filter.md), [`Polygon Visualization`](polygon_visualization.md), [`Moondream2`](moondream2.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Background Color Visualization`](background_color_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`EasyOCR`](easy_ocr.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Corner Visualization`](corner_visualization.md), [`Motion Detection`](motion_detection.md), [`SORT Tracker`](sort_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Time in Zone`](timein_zone.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Depth Estimation`](depth_estimation.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Time in Zone`](timein_zone.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Mask Area Measurement`](mask_area_measurement.md), [`SIFT Comparison`](sift_comparison.md), [`Image Slicer`](image_slicer.md), [`Image Stack`](image_stack.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Line Counter`](line_counter.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stitch Images`](stitch_images.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Detections Stitch`](detections_stitch.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Byte Tracker`](byte_tracker.md), [`OCR Model`](ocr_model.md), [`Perspective Correction`](perspective_correction.md), [`Crop Visualization`](crop_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Image Slicer`](image_slicer.md), [`Camera Focus`](camera_focus.md), [`Identify Changes`](identify_changes.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Line Counter`](line_counter.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Distance Measurement`](distance_measurement.md), [`Detections Consensus`](detections_consensus.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Pixel Color Count`](pixel_color_count.md), [`Trace Visualization`](trace_visualization.md), [`SAM 3`](sam3.md), [`Circle Visualization`](circle_visualization.md), [`Time in Zone`](timein_zone.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Image Contours`](image_contours.md), [`Path Deviation`](path_deviation.md), [`Contrast Equalization`](contrast_equalization.md)
    - outputs: [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Triangle Visualization`](triangle_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Detections Combine`](detections_combine.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Detections Merge`](detections_merge.md), [`Florence-2 Model`](florence2_model.md), [`Mask Visualization`](mask_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Detections Transformation`](detections_transformation.md), [`Camera Focus`](camera_focus.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Halo Visualization`](halo_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Track Class Lock`](track_class_lock.md), [`Dynamic Crop`](dynamic_crop.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Velocity`](velocity.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Halo Visualization`](halo_visualization.md), [`Label Visualization`](label_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Detection Offset`](detection_offset.md), [`Path Deviation`](path_deviation.md), [`Florence-2 Model`](florence2_model.md), [`Overlap Filter`](overlap_filter.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Detections Filter`](detections_filter.md), [`Polygon Visualization`](polygon_visualization.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Background Color Visualization`](background_color_visualization.md), [`Overlap Analysis`](overlap_analysis.md), [`Corner Visualization`](corner_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Time in Zone`](timein_zone.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Time in Zone`](timein_zone.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Polygon Visualization`](polygon_visualization.md), [`Line Counter`](line_counter.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Byte Tracker`](byte_tracker.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Perspective Correction`](perspective_correction.md), [`Crop Visualization`](crop_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Line Counter`](line_counter.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Distance Measurement`](distance_measurement.md), [`Detections Consensus`](detections_consensus.md), [`Size Measurement`](size_measurement.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Time in Zone`](timein_zone.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Event Writer`](event_writer.md), [`Path Deviation`](path_deviation.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`SAM2 Video Tracker` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `boxes` (*Union[[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md)]*): Bounding boxes to use as SAM2 prompts.  Only read on frames where the block re-prompts (see `prompt_mode`)..
        - `model_id` (*[`roboflow_model_id`](../kinds/roboflow_model_id.md)*): Streaming video tracker model id resolved by `inference_models`.  The `sam2video` family ships four Hiera backbone sizes; `small` is the default trade-off between speed and quality.  `sam3trackervideo` is SAM3's visually prompted tracker — the same prompt contract with a larger backbone, markedly better at identity retention on long videos and crowded scenes, at higher compute cost..
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


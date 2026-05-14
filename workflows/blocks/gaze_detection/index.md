
# Gaze Detection

!!! warning "Deprecated"

    This block is deprecated and may be removed in a future release.



??? "Class: `GazeBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/models/foundation/gaze/v1.py">inference.core.workflows.core_steps.models.foundation.gaze.v1.GazeBlockV1</a>
    



**DEPRECATED.** L2CS Gaze detection has been removed from inference along
with the MediaPipe dependency. Invoking this block raises
`FeatureDeprecatedError` (HTTP 410 Gone).


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/gaze@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `do_run_face_detection` | `bool` | Whether to run face detection. Set to False if input images are pre-cropped face images.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Gaze Detection` in version `v1`.

    - inputs: [`Triangle Visualization`](triangle_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Stitch Images`](stitch_images.md), [`Ellipse Visualization`](ellipse_visualization.md), [`S3 Sink`](s3_sink.md), [`Image Blur`](image_blur.md), [`Background Subtraction`](background_subtraction.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Camera Focus`](camera_focus.md), [`Morphological Transformation`](morphological_transformation.md), [`Identify Changes`](identify_changes.md), [`QR Code Generator`](qr_code_generator.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Image Preprocessing`](image_preprocessing.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Email Notification`](email_notification.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Crop Visualization`](crop_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Relative Static Crop`](relative_static_crop.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Identify Outliers`](identify_outliers.md), [`Perspective Correction`](perspective_correction.md), [`Halo Visualization`](halo_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Email Notification`](email_notification.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Contours`](image_contours.md), [`Text Display`](text_display.md), [`Icon Visualization`](icon_visualization.md), [`SIFT`](sift.md), [`Circle Visualization`](circle_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Image Slicer`](image_slicer.md), [`Camera Focus`](camera_focus.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Image Slicer`](image_slicer.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Motion Detection`](motion_detection.md), [`Polygon Visualization`](polygon_visualization.md), [`JSON Parser`](json_parser.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Image Threshold`](image_threshold.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Grid Visualization`](grid_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Color Visualization`](color_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Label Visualization`](label_visualization.md), [`Detections Consensus`](detections_consensus.md), [`VLM As Detector`](vlm_as_detector.md), [`Dot Visualization`](dot_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Morphological Transformation`](morphological_transformation.md), [`Trace Visualization`](trace_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Slack Notification`](slack_notification.md), [`VLM As Classifier`](vlm_as_classifier.md), [`SIFT Comparison`](sift_comparison.md), [`Depth Estimation`](depth_estimation.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Local File Sink`](local_file_sink.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Webhook Sink`](webhook_sink.md)
    - outputs: [`Triangle Visualization`](triangle_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SAM 3`](sam3.md), [`Detections Transformation`](detections_transformation.md), [`Dynamic Zone`](dynamic_zone.md), [`Florence-2 Model`](florence2_model.md), [`SORT Tracker`](sort_tracker.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Distance Measurement`](distance_measurement.md), [`OpenAI`](open_ai.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Webhook Sink`](webhook_sink.md), [`Google Gemini`](google_gemini.md), [`Detection Event Log`](detection_event_log.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Crop Visualization`](crop_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Seg Preview`](seg_preview.md), [`Continue If`](continue_if.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Text Display`](text_display.md), [`Icon Visualization`](icon_visualization.md), [`Circle Visualization`](circle_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenAI`](open_ai.md), [`Velocity`](velocity.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`SAM 3`](sam3.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Google Gemini`](google_gemini.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Dynamic Crop`](dynamic_crop.md), [`Color Visualization`](color_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Label Visualization`](label_visualization.md), [`Detections Consensus`](detections_consensus.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detections Merge`](detections_merge.md), [`Dot Visualization`](dot_visualization.md), [`Detection Offset`](detection_offset.md), [`Camera Calibration`](camera_calibration.md), [`SAM 3`](sam3.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Trace Visualization`](trace_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Anthropic Claude`](anthropic_claude.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Google Gemma API`](google_gemma_api.md), [`Detections Filter`](detections_filter.md), [`Template Matching`](template_matching.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Line Counter Visualization`](line_counter_visualization.md), [`OpenAI`](open_ai.md), [`Byte Tracker`](byte_tracker.md), [`Mask Edge Snap`](mask_edge_snap.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Gaze Detection` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): The image to infer on..
        - `do_run_face_detection` (*[`boolean`](../kinds/boolean.md)*): Whether to run face detection. Set to False if input images are pre-cropped face images..

    - output
    
        - `face_predictions` ([`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md)): Prediction with detected bounding boxes and detected keypoints in form of sv.Detections(...) object.
        - `yaw_degrees` ([`float`](../kinds/float.md)): Float value.
        - `pitch_degrees` ([`float`](../kinds/float.md)): Float value.



??? tip "Example JSON definition of step `Gaze Detection` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/gaze@v1",
	    "images": "$inputs.image",
	    "do_run_face_detection": "<block_does_not_provide_example>"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    


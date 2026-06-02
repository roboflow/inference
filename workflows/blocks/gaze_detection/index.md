
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

    - inputs: [`Bounding Box Visualization`](bounding_box_visualization.md), [`Image Slicer`](image_slicer.md), [`Identify Outliers`](identify_outliers.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Morphological Transformation`](morphological_transformation.md), [`Slack Notification`](slack_notification.md), [`Camera Calibration`](camera_calibration.md), [`Email Notification`](email_notification.md), [`Corner Visualization`](corner_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Motion Detection`](motion_detection.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`VLM As Detector`](vlm_as_detector.md), [`Color Visualization`](color_visualization.md), [`Text Display`](text_display.md), [`Line Counter Visualization`](line_counter_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Slicer`](image_slicer.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Email Notification`](email_notification.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Camera Focus`](camera_focus.md), [`Image Blur`](image_blur.md), [`Dynamic Crop`](dynamic_crop.md), [`Identify Changes`](identify_changes.md), [`Relative Static Crop`](relative_static_crop.md), [`Halo Visualization`](halo_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Perspective Correction`](perspective_correction.md), [`Camera Focus`](camera_focus.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Trace Visualization`](trace_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`JSON Parser`](json_parser.md), [`Dot Visualization`](dot_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Crop Visualization`](crop_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Detections Consensus`](detections_consensus.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Circle Visualization`](circle_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Grid Visualization`](grid_visualization.md), [`Image Threshold`](image_threshold.md), [`Triangle Visualization`](triangle_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Local File Sink`](local_file_sink.md), [`Image Preprocessing`](image_preprocessing.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Depth Estimation`](depth_estimation.md), [`SIFT Comparison`](sift_comparison.md), [`Webhook Sink`](webhook_sink.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Mask Visualization`](mask_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stitch Images`](stitch_images.md), [`SIFT`](sift.md), [`Icon Visualization`](icon_visualization.md), [`Label Visualization`](label_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Contrast Equalization`](contrast_equalization.md), [`S3 Sink`](s3_sink.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Image Contours`](image_contours.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Blur Visualization`](blur_visualization.md)
    - outputs: [`Anthropic Claude`](anthropic_claude.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Distance Measurement`](distance_measurement.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`OpenAI`](open_ai.md), [`SAM 3`](sam3.md), [`SAM 3`](sam3.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Camera Calibration`](camera_calibration.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Google Gemma`](google_gemma.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Corner Visualization`](corner_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Color Visualization`](color_visualization.md), [`OpenRouter`](open_router.md), [`Text Display`](text_display.md), [`Line Counter Visualization`](line_counter_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Continue If`](continue_if.md), [`Ellipse Visualization`](ellipse_visualization.md), [`SORT Tracker`](sort_tracker.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Florence-2 Model`](florence2_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Google Gemma API`](google_gemma_api.md), [`Seg Preview`](seg_preview.md), [`Trace Visualization`](trace_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Dot Visualization`](dot_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Qwen-VL`](qwen_vl.md), [`Background Color Visualization`](background_color_visualization.md), [`Crop Visualization`](crop_visualization.md), [`SAM 3`](sam3.md), [`OpenAI`](open_ai.md), [`Detections Consensus`](detections_consensus.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Detections Merge`](detections_merge.md), [`Detections Transformation`](detections_transformation.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Detections Filter`](detections_filter.md), [`Google Gemini`](google_gemini.md), [`Circle Visualization`](circle_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Google Gemini`](google_gemini.md), [`Triangle Visualization`](triangle_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenAI`](open_ai.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Velocity`](velocity.md), [`Detection Offset`](detection_offset.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Webhook Sink`](webhook_sink.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Template Matching`](template_matching.md), [`Google Gemini`](google_gemini.md), [`Icon Visualization`](icon_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Label Visualization`](label_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Blur Visualization`](blur_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md)

    
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


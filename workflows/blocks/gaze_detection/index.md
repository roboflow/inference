
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

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-shield-alert:{ style="color: #d32f2f" } `hard` — runtime `self_hosted_cpu`; execution `local`
:   Requires a GPU; run_locally() loads a model that needs CUDA.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Gaze Detection` in version `v1`.

    - inputs: [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Camera Focus`](camera_focus.md), [`Halo Visualization`](halo_visualization.md), [`Event Writer`](event_writer.md), [`Email Notification`](email_notification.md), [`QR Code Generator`](qr_code_generator.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Local File Sink`](local_file_sink.md), [`Camera Focus`](camera_focus.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Image Contours`](image_contours.md), [`Image Preprocessing`](image_preprocessing.md), [`Background Subtraction`](background_subtraction.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Color Visualization`](color_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Image Slicer`](image_slicer.md), [`Depth Estimation`](depth_estimation.md), [`Text Display`](text_display.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`JSON Parser`](json_parser.md), [`Relative Static Crop`](relative_static_crop.md), [`Icon Visualization`](icon_visualization.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`SIFT Comparison`](sift_comparison.md), [`Grid Visualization`](grid_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Image Threshold`](image_threshold.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Label Visualization`](label_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Morphological Transformation`](morphological_transformation.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Image Blur`](image_blur.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Identify Changes`](identify_changes.md), [`MQTT Writer`](mqtt_writer.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Dynamic Crop`](dynamic_crop.md), [`Camera Calibration`](camera_calibration.md), [`Polygon Visualization`](polygon_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`SIFT Comparison`](sift_comparison.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Perspective Correction`](perspective_correction.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Stitch Images`](stitch_images.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Contrast Equalization`](contrast_equalization.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Background Color Visualization`](background_color_visualization.md), [`Detections Consensus`](detections_consensus.md), [`Corner Visualization`](corner_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Identify Outliers`](identify_outliers.md), [`Dot Visualization`](dot_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Email Notification`](email_notification.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Slack Notification`](slack_notification.md), [`S3 Sink`](s3_sink.md), [`Webhook Sink`](webhook_sink.md), [`SIFT`](sift.md), [`Heatmap Visualization`](heatmap_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Image Slicer`](image_slicer.md)
    - outputs: [`Google Gemma API`](google_gemma_api.md), [`Event Writer`](event_writer.md), [`Google Gemini`](google_gemini.md), [`Ellipse Visualization`](ellipse_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`SAM 3`](sam3.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Color Visualization`](color_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Template Matching`](template_matching.md), [`Seg Preview`](seg_preview.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Text Display`](text_display.md), [`Distance Measurement`](distance_measurement.md), [`Icon Visualization`](icon_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`OpenRouter`](open_router.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Google Gemini`](google_gemini.md), [`Anthropic Claude`](anthropic_claude.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SAM 3`](sam3.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Google Gemma`](google_gemma.md), [`Label Visualization`](label_visualization.md), [`Google Gemini`](google_gemini.md), [`Keypoint Visualization`](keypoint_visualization.md), [`MQTT Writer`](mqtt_writer.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Dynamic Crop`](dynamic_crop.md), [`Camera Calibration`](camera_calibration.md), [`Florence-2 Model`](florence2_model.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Detections Transformation`](detections_transformation.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Byte Tracker`](byte_tracker.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Florence-2 Model`](florence2_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`SAM 3`](sam3.md), [`Detection Offset`](detection_offset.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Detections Merge`](detections_merge.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`SORT Tracker`](sort_tracker.md), [`Qwen-VL`](qwen_vl.md), [`Triangle Visualization`](triangle_visualization.md), [`Detections Consensus`](detections_consensus.md), [`Background Color Visualization`](background_color_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`OpenAI`](open_ai.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Continue If`](continue_if.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detection Event Log`](detection_event_log.md), [`Webhook Sink`](webhook_sink.md), [`OpenAI`](open_ai.md), [`Detections Filter`](detections_filter.md), [`Heatmap Visualization`](heatmap_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Velocity`](velocity.md)

    
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



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

    - inputs: [`Circle Visualization`](circle_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Image Blur`](image_blur.md), [`S3 Sink`](s3_sink.md), [`Email Notification`](email_notification.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Camera Focus`](camera_focus.md), [`SIFT Comparison`](sift_comparison.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Event Writer`](event_writer.md), [`Identify Changes`](identify_changes.md), [`Slack Notification`](slack_notification.md), [`Halo Visualization`](halo_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Dot Visualization`](dot_visualization.md), [`Image Slicer`](image_slicer.md), [`Label Visualization`](label_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Email Notification`](email_notification.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`JSON Parser`](json_parser.md), [`Stitch Images`](stitch_images.md), [`Identify Outliers`](identify_outliers.md), [`Morphological Transformation`](morphological_transformation.md), [`Blur Visualization`](blur_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Camera Focus`](camera_focus.md), [`Corner Visualization`](corner_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`MQTT Writer`](mqtt_writer.md), [`SIFT Comparison`](sift_comparison.md), [`Webhook Sink`](webhook_sink.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Image Threshold`](image_threshold.md), [`Image Contours`](image_contours.md), [`Local File Sink`](local_file_sink.md), [`Motion Detection`](motion_detection.md), [`Polygon Visualization`](polygon_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`SIFT`](sift.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`Grid Visualization`](grid_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Keypoint Visualization`](keypoint_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Icon Visualization`](icon_visualization.md), [`Color Visualization`](color_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Triangle Visualization`](triangle_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Dynamic Crop`](dynamic_crop.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Background Subtraction`](background_subtraction.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Crop Visualization`](crop_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Mask Visualization`](mask_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Image Slicer`](image_slicer.md), [`Perspective Correction`](perspective_correction.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Text Display`](text_display.md), [`Morphological Transformation`](morphological_transformation.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`VLM As Detector`](vlm_as_detector.md), [`Depth Estimation`](depth_estimation.md), [`Detections Consensus`](detections_consensus.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Contrast Equalization`](contrast_equalization.md), [`Camera Calibration`](camera_calibration.md), [`Ellipse Visualization`](ellipse_visualization.md), [`VLM As Detector`](vlm_as_detector.md)
    - outputs: [`SAM 3`](sam3.md), [`Circle Visualization`](circle_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Event Writer`](event_writer.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Google Gemma`](google_gemma.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Dot Visualization`](dot_visualization.md), [`SAM 3`](sam3.md), [`Label Visualization`](label_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Velocity`](velocity.md), [`Pixelate Visualization`](pixelate_visualization.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Google Gemini`](google_gemini.md), [`Track Class Lock`](track_class_lock.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Gemma API`](google_gemma_api.md), [`OpenAI`](open_ai.md), [`Trace Visualization`](trace_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`Google Gemini`](google_gemini.md), [`Corner Visualization`](corner_visualization.md), [`OpenRouter`](open_router.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`MQTT Writer`](mqtt_writer.md), [`Webhook Sink`](webhook_sink.md), [`SAM 3`](sam3.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Byte Tracker`](byte_tracker.md), [`Google Gemini`](google_gemini.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Template Matching`](template_matching.md), [`Anthropic Claude`](anthropic_claude.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Icon Visualization`](icon_visualization.md), [`Color Visualization`](color_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Triangle Visualization`](triangle_visualization.md), [`Seg Preview`](seg_preview.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Dynamic Crop`](dynamic_crop.md), [`Detections Transformation`](detections_transformation.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Crop Visualization`](crop_visualization.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Continue If`](continue_if.md), [`OpenAI`](open_ai.md), [`Detection Offset`](detection_offset.md), [`Florence-2 Model`](florence2_model.md), [`SORT Tracker`](sort_tracker.md), [`Distance Measurement`](distance_measurement.md), [`Qwen-VL`](qwen_vl.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Anthropic Claude`](anthropic_claude.md), [`Text Display`](text_display.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Detections Consensus`](detections_consensus.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Detections Filter`](detections_filter.md), [`Camera Calibration`](camera_calibration.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Detections Merge`](detections_merge.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`SAM3 Video Tracker`](sam3_video_tracker.md)

    
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


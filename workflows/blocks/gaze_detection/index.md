
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

    - inputs: [`Image Blur`](image_blur.md), [`VLM As Detector`](vlm_as_detector.md), [`Image Threshold`](image_threshold.md), [`Morphological Transformation`](morphological_transformation.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Triangle Visualization`](triangle_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Slack Notification`](slack_notification.md), [`Email Notification`](email_notification.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Grid Visualization`](grid_visualization.md), [`Mask Visualization`](mask_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Camera Focus`](camera_focus.md), [`Dynamic Zone`](dynamic_zone.md), [`Halo Visualization`](halo_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Webhook Sink`](webhook_sink.md), [`Dynamic Crop`](dynamic_crop.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`QR Code Generator`](qr_code_generator.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Halo Visualization`](halo_visualization.md), [`Label Visualization`](label_visualization.md), [`SIFT`](sift.md), [`Morphological Transformation`](morphological_transformation.md), [`Text Display`](text_display.md), [`MQTT Writer`](mqtt_writer.md), [`PLC Writer`](plc_writer.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`PLC Reader`](plc_reader.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Corner Visualization`](corner_visualization.md), [`Motion Detection`](motion_detection.md), [`Identify Outliers`](identify_outliers.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Depth Estimation`](depth_estimation.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Classification Label Visualization`](classification_label_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Image Slicer`](image_slicer.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stitch Images`](stitch_images.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Local File Sink`](local_file_sink.md), [`Relative Static Crop`](relative_static_crop.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Perspective Correction`](perspective_correction.md), [`Crop Visualization`](crop_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Image Slicer`](image_slicer.md), [`Camera Focus`](camera_focus.md), [`Identify Changes`](identify_changes.md), [`JSON Parser`](json_parser.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Email Notification`](email_notification.md), [`S3 Sink`](s3_sink.md), [`Detections Consensus`](detections_consensus.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Event Writer`](event_writer.md), [`Image Contours`](image_contours.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Contrast Equalization`](contrast_equalization.md)
    - outputs: [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Byte Tracker`](byte_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Triangle Visualization`](triangle_visualization.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Qwen-VL`](qwen_vl.md), [`Pixelate Visualization`](pixelate_visualization.md), [`OpenRouter`](open_router.md), [`Mask Edge Snap`](mask_edge_snap.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Detections Merge`](detections_merge.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Detections Transformation`](detections_transformation.md), [`OpenAI`](open_ai.md), [`Template Matching`](template_matching.md), [`Track Class Lock`](track_class_lock.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Dynamic Zone`](dynamic_zone.md), [`Anthropic Claude`](anthropic_claude.md), [`Webhook Sink`](webhook_sink.md), [`SAM 3`](sam3.md), [`Dynamic Crop`](dynamic_crop.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Velocity`](velocity.md), [`SAM 3`](sam3.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Color Visualization`](color_visualization.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Label Visualization`](label_visualization.md), [`Detection Offset`](detection_offset.md), [`Florence-2 Model`](florence2_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Seg Preview`](seg_preview.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Text Display`](text_display.md), [`MQTT Writer`](mqtt_writer.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Google Gemini`](google_gemini.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Google Gemma API`](google_gemma_api.md), [`Detections Filter`](detections_filter.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Background Color Visualization`](background_color_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Google Gemini`](google_gemini.md), [`Corner Visualization`](corner_visualization.md), [`SORT Tracker`](sort_tracker.md), [`OpenAI`](open_ai.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Google Gemma`](google_gemma.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Crop Visualization`](crop_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Distance Measurement`](distance_measurement.md), [`Detections Consensus`](detections_consensus.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Trace Visualization`](trace_visualization.md), [`SAM 3`](sam3.md), [`Circle Visualization`](circle_visualization.md), [`Event Writer`](event_writer.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Detection Event Log`](detection_event_log.md), [`Continue If`](continue_if.md), [`Stitch OCR Detections`](stitch_ocr_detections.md)

    
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


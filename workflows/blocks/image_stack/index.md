
# Image Stack



??? "Class: `ImageStackBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/fusion/image_stack/v1.py">inference.core.workflows.core_steps.fusion.image_stack.v1.ImageStackBlockV1</a>
    



Accumulate compressed video frames into a fixed-size stack, returning the most recent
N frames as JPEG-encoded binary blobs. Designed for shared-hosting safety: frames are
always JPEG-compressed and downsampled to fit within resolution limits, preventing
out-of-memory conditions.

## How This Block Works

1. Receives a video frame (WorkflowImageData) each workflow cycle.
2. Downsamples the frame if it exceeds the configured resolution limits (default
   1920x1080), preserving aspect ratio.
3. JPEG-encodes the frame at quality 75 and stores the resulting bytes.
4. Maintains a per-camera FIFO buffer (deque) of up to `stack_size` compressed frames.
   When the buffer is full the oldest frame is automatically evicted.
5. If `stack_size` changes between calls (e.g. via a dynamic selector), the buffer is
   resized and existing frames are preserved up to the new limit.
6. If the `clear` input is True the buffer is flushed before the current frame is added.
7. Outputs the list of JPEG byte blobs (newest first) and the current frame count.

## Common Use Cases

- **Action / activity recognition**: accumulate a clip of N frames and pass them to a
  vision-language model (e.g. Google Gemini, Qwen) that can reason over multiple images
  to classify actions, detect events, or describe what is happening in a scene.
- **Time-lapse snapshots**: collect the last N frames for periodic visual comparison.
- **Event buffering**: keep a rolling window of frames around an event of interest.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/image_stack@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `stack_size` | `int` | Maximum number of frames to keep in the stack (1-64). When the stack is full the oldest frame is evicted.. | ✅ |
| `resolution_width` | `int` | Maximum frame width in pixels (64-1920). Frames wider than this are downsampled preserving aspect ratio.. | ✅ |
| `resolution_height` | `int` | Maximum frame height in pixels (64-1080). Frames taller than this are downsampled preserving aspect ratio.. | ✅ |
| `clear` | `bool` | When True the entire frame buffer is flushed before the current frame is added. Useful for resetting state on scene changes.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-alert-circle-outline:{ style="color: #f57c00" } `soft` — runtime `hosted_serverless`, `dedicated_deployment`; execution `remote`; input `video`
:   Frame stack is stored in process memory per video_identifier. With remote step execution on stateless or multi-replica HTTP runtimes, successive frames may be served by different worker processes, so the stack resets or contains only a partial frame history. Use local step execution in an InferencePipeline for stable cross-frame results.

:material-alert-circle-outline:{ style="color: #f57c00" } `soft` — input `image`
:   Block depends on temporal context from video or repeated-frame workflows. With a still image/photo, there is no meaningful history to track, compare, aggregate, or visualize, so the block provides little or no benefit.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Image Stack` in version `v1`.

    - inputs: [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Dynamic Zone`](dynamic_zone.md), [`Slack Notification`](slack_notification.md), [`MQTT Writer`](mqtt_writer.md), [`Detections Consensus`](detections_consensus.md), [`JSON Parser`](json_parser.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Threshold`](image_threshold.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Slicer`](image_slicer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Image Blur`](image_blur.md), [`Depth Estimation`](depth_estimation.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Camera Focus`](camera_focus.md), [`Email Notification`](email_notification.md), [`Crop Visualization`](crop_visualization.md), [`Label Visualization`](label_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Text Display`](text_display.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Identify Outliers`](identify_outliers.md), [`Corner Visualization`](corner_visualization.md), [`Image Slicer`](image_slicer.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Distance Measurement`](distance_measurement.md), [`Image Preprocessing`](image_preprocessing.md), [`Morphological Transformation`](morphological_transformation.md), [`Triangle Visualization`](triangle_visualization.md), [`Identify Changes`](identify_changes.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`PLC Reader`](plc_reader.md), [`Halo Visualization`](halo_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Background Subtraction`](background_subtraction.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Motion Detection`](motion_detection.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Detection Event Log`](detection_event_log.md), [`SIFT`](sift.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Line Counter`](line_counter.md), [`SIFT Comparison`](sift_comparison.md), [`Grid Visualization`](grid_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Template Matching`](template_matching.md), [`PLC Writer`](plc_writer.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Stitch Images`](stitch_images.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Line Counter`](line_counter.md), [`Circle Visualization`](circle_visualization.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Event Writer`](event_writer.md), [`Image Stack`](image_stack.md), [`Relative Static Crop`](relative_static_crop.md), [`VLM As Detector`](vlm_as_detector.md), [`QR Code Generator`](qr_code_generator.md), [`Dot Visualization`](dot_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`S3 Sink`](s3_sink.md), [`Pixel Color Count`](pixel_color_count.md), [`Camera Focus`](camera_focus.md), [`Color Visualization`](color_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Blur Visualization`](blur_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Dynamic Crop`](dynamic_crop.md), [`Halo Visualization`](halo_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Morphological Transformation`](morphological_transformation.md), [`Polygon Visualization`](polygon_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Local File Sink`](local_file_sink.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Email Notification`](email_notification.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Image Contours`](image_contours.md), [`Roboflow Vision Events`](roboflow_vision_events.md)
    - outputs: [`YOLO-World Model`](yolo_world_model.md), [`SAM 3`](sam3.md), [`Dynamic Zone`](dynamic_zone.md), [`Byte Tracker`](byte_tracker.md), [`SORT Tracker`](sort_tracker.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Slack Notification`](slack_notification.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Image Slicer`](image_slicer.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Pixelate Visualization`](pixelate_visualization.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Identify Outliers`](identify_outliers.md), [`Corner Visualization`](corner_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`PLC Reader`](plc_reader.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`VLM As Detector`](vlm_as_detector.md), [`Background Subtraction`](background_subtraction.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Motion Detection`](motion_detection.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`SIFT Comparison`](sift_comparison.md), [`Size Measurement`](size_measurement.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Gemini`](google_gemini.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Stitch Images`](stitch_images.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Circle Visualization`](circle_visualization.md), [`OpenAI`](open_ai.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Image Stack`](image_stack.md), [`Path Deviation`](path_deviation.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`VLM As Detector`](vlm_as_detector.md), [`QR Code Generator`](qr_code_generator.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Mask Visualization`](mask_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Perspective Correction`](perspective_correction.md), [`SIFT Comparison`](sift_comparison.md), [`Trace Visualization`](trace_visualization.md), [`Blur Visualization`](blur_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Dominant Color`](dominant_color.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Florence-2 Model`](florence2_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Email Notification`](email_notification.md), [`Image Contours`](image_contours.md), [`Qwen-VL`](qwen_vl.md), [`Florence-2 Model`](florence2_model.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Clip Comparison`](clip_comparison.md), [`Cache Set`](cache_set.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`MQTT Writer`](mqtt_writer.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Detections Consensus`](detections_consensus.md), [`Time in Zone`](timein_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`Byte Tracker`](byte_tracker.md), [`Image Threshold`](image_threshold.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Image Blur`](image_blur.md), [`Google Gemini`](google_gemini.md), [`Email Notification`](email_notification.md), [`Crop Visualization`](crop_visualization.md), [`Time in Zone`](timein_zone.md), [`Label Visualization`](label_visualization.md), [`Track Class Lock`](track_class_lock.md), [`Text Display`](text_display.md), [`Buffer`](buffer.md), [`Image Slicer`](image_slicer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemma API`](google_gemma_api.md), [`Triangle Visualization`](triangle_visualization.md), [`SAM 3`](sam3.md), [`Image Preprocessing`](image_preprocessing.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Identify Changes`](identify_changes.md), [`Halo Visualization`](halo_visualization.md), [`Path Deviation`](path_deviation.md), [`OpenRouter`](open_router.md), [`Byte Tracker`](byte_tracker.md), [`Line Counter`](line_counter.md), [`Grid Visualization`](grid_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Seg Preview`](seg_preview.md), [`Google Gemma`](google_gemma.md), [`Line Counter`](line_counter.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Object Detection Model`](object_detection_model.md), [`Event Writer`](event_writer.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Detection Offset`](detection_offset.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Color Visualization`](color_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Webhook Sink`](webhook_sink.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`OpenAI`](open_ai.md), [`Icon Visualization`](icon_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`SAM2 Video Tracker`](sam2_video_tracker.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Image Stack` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Video frame to add to the stack..
        - `stack_size` (*[`integer`](../kinds/integer.md)*): Maximum number of frames to keep in the stack (1-64). When the stack is full the oldest frame is evicted..
        - `resolution_width` (*[`integer`](../kinds/integer.md)*): Maximum frame width in pixels (64-1920). Frames wider than this are downsampled preserving aspect ratio..
        - `resolution_height` (*[`integer`](../kinds/integer.md)*): Maximum frame height in pixels (64-1080). Frames taller than this are downsampled preserving aspect ratio..
        - `clear` (*[`boolean`](../kinds/boolean.md)*): When True the entire frame buffer is flushed before the current frame is added. Useful for resetting state on scene changes..

    - output
    
        - `frames` ([`list_of_values`](../kinds/list_of_values.md)): List of values of any type.
        - `frames_count` ([`integer`](../kinds/integer.md)): Integer value.



??? tip "Example JSON definition of step `Image Stack` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/image_stack@v1",
	    "image": "$inputs.image",
	    "stack_size": 5,
	    "resolution_width": 640,
	    "resolution_height": 480,
	    "clear": false
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    


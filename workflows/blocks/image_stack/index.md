
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

    - inputs: [`PLC Writer`](plc_writer.md), [`Image Blur`](image_blur.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Blur Visualization`](blur_visualization.md), [`Frame Delay`](frame_delay.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Crop Visualization`](crop_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Polygon Visualization`](polygon_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Event Writer`](event_writer.md), [`Image Slicer`](image_slicer.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Email Notification`](email_notification.md), [`Webhook Sink`](webhook_sink.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Template Matching`](template_matching.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Perspective Correction`](perspective_correction.md), [`Motion Detection`](motion_detection.md), [`Halo Visualization`](halo_visualization.md), [`SIFT`](sift.md), [`Slack Notification`](slack_notification.md), [`Label Visualization`](label_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`PLC Reader`](plc_reader.md), [`Detections Consensus`](detections_consensus.md), [`Label Visualization`](label_visualization.md), [`S3 Sink`](s3_sink.md), [`Halo Visualization`](halo_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Email Notification`](email_notification.md), [`Detection Event Log`](detection_event_log.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Pixel Color Count`](pixel_color_count.md), [`Background Subtraction`](background_subtraction.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Stitch Images`](stitch_images.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Relative Static Crop`](relative_static_crop.md), [`Corner Visualization`](corner_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Depth Estimation`](depth_estimation.md), [`SIFT Comparison`](sift_comparison.md), [`JSON Parser`](json_parser.md), [`Distance Measurement`](distance_measurement.md), [`Dot Visualization`](dot_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Background Color Visualization`](background_color_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Image Contours`](image_contours.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Camera Focus`](camera_focus.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Contrast Equalization`](contrast_equalization.md), [`Morphological Transformation`](morphological_transformation.md), [`SIFT Comparison`](sift_comparison.md), [`Image Threshold`](image_threshold.md), [`Trace Visualization`](trace_visualization.md), [`Color Visualization`](color_visualization.md), [`Camera Focus`](camera_focus.md), [`Line Counter`](line_counter.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Local File Sink`](local_file_sink.md), [`Circle Visualization`](circle_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Morphological Transformation`](morphological_transformation.md), [`Dynamic Zone`](dynamic_zone.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`Polygon Visualization`](polygon_visualization.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`MQTT Writer`](mqtt_writer.md), [`Icon Visualization`](icon_visualization.md), [`Rich Label Visualization`](rich_label_visualization.md), [`Identify Changes`](identify_changes.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`Image Slicer`](image_slicer.md), [`Line Counter`](line_counter.md), [`Text Display`](text_display.md), [`Image Stack`](image_stack.md), [`Grid Visualization`](grid_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Identify Outliers`](identify_outliers.md), [`Mask Visualization`](mask_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`VLM As Detector`](vlm_as_detector.md)
    - outputs: [`Track Class Lock`](track_class_lock.md), [`Image Blur`](image_blur.md), [`Byte Tracker`](byte_tracker.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Path Deviation`](path_deviation.md), [`Crop Visualization`](crop_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Object Detection Model`](object_detection_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Image Slicer`](image_slicer.md), [`Webhook Sink`](webhook_sink.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Motion Detection`](motion_detection.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Clip Comparison`](clip_comparison.md), [`YOLO-World Model`](yolo_world_model.md), [`Buffer`](buffer.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Slack Notification`](slack_notification.md), [`Label Visualization`](label_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`PLC Reader`](plc_reader.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Label Visualization`](label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Email Notification`](email_notification.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Pixel Color Count`](pixel_color_count.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Path Deviation`](path_deviation.md), [`Background Subtraction`](background_subtraction.md), [`Stitch Images`](stitch_images.md), [`Corner Visualization`](corner_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Detection Offset`](detection_offset.md), [`Seg Preview`](seg_preview.md), [`Google Gemini`](google_gemini.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Trace Visualization`](trace_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Object Detection Model`](object_detection_model.md), [`Color Visualization`](color_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Dominant Color`](dominant_color.md), [`Morphological Transformation`](morphological_transformation.md), [`Dynamic Zone`](dynamic_zone.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Google Gemma API`](google_gemma_api.md), [`Google Gemma`](google_gemma.md), [`Polygon Visualization`](polygon_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Rich Label Visualization`](rich_label_visualization.md), [`Identify Changes`](identify_changes.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Keypoint Visualization`](keypoint_visualization.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Clip Comparison`](clip_comparison.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Object Detection Model`](object_detection_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Mask Visualization`](mask_visualization.md), [`Nearest Neighbor Detection Match`](nearest_neighbor_detection_match.md), [`OpenAI`](open_ai.md), [`Byte Tracker`](byte_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`VLM As Detector`](vlm_as_detector.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Blur Visualization`](blur_visualization.md), [`Frame Delay`](frame_delay.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Reference Path Visualization`](reference_path_visualization.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Gemini`](google_gemini.md), [`Event Writer`](event_writer.md), [`Cache Set`](cache_set.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Time in Zone`](timein_zone.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Email Notification`](email_notification.md), [`Perspective Correction`](perspective_correction.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Halo Visualization`](halo_visualization.md), [`OpenRouter`](open_router.md), [`Detections Consensus`](detections_consensus.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Halo Visualization`](halo_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Time in Zone`](timein_zone.md), [`Time in Zone`](timein_zone.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Size Measurement`](size_measurement.md), [`SIFT Comparison`](sift_comparison.md), [`SAM 3`](sam3.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Dot Visualization`](dot_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`Image Contours`](image_contours.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Morphological Transformation`](morphological_transformation.md), [`Image Threshold`](image_threshold.md), [`SAM 3`](sam3.md), [`Line Counter`](line_counter.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Circle Visualization`](circle_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Google Gemini`](google_gemini.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`MQTT Writer`](mqtt_writer.md), [`SAM 3`](sam3.md), [`Anthropic Claude`](anthropic_claude.md), [`SORT Tracker`](sort_tracker.md), [`Image Preprocessing`](image_preprocessing.md), [`Line Counter`](line_counter.md), [`Anthropic Claude`](anthropic_claude.md), [`Text Display`](text_display.md), [`Image Stack`](image_stack.md), [`Grid Visualization`](grid_visualization.md), [`Florence-2 Model`](florence2_model.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Identify Outliers`](identify_outliers.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Pixelate Visualization`](pixelate_visualization.md)

    
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


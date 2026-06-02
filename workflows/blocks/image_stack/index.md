
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

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Image Stack` in version `v1`.

    - inputs: [`Bounding Box Visualization`](bounding_box_visualization.md), [`Image Slicer`](image_slicer.md), [`Distance Measurement`](distance_measurement.md), [`Identify Outliers`](identify_outliers.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Morphological Transformation`](morphological_transformation.md), [`Slack Notification`](slack_notification.md), [`Camera Calibration`](camera_calibration.md), [`Email Notification`](email_notification.md), [`Corner Visualization`](corner_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Motion Detection`](motion_detection.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`VLM As Detector`](vlm_as_detector.md), [`Color Visualization`](color_visualization.md), [`Text Display`](text_display.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Line Counter`](line_counter.md), [`Image Slicer`](image_slicer.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Email Notification`](email_notification.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Camera Focus`](camera_focus.md), [`Image Blur`](image_blur.md), [`Dynamic Crop`](dynamic_crop.md), [`Identify Changes`](identify_changes.md), [`Relative Static Crop`](relative_static_crop.md), [`Halo Visualization`](halo_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Perspective Correction`](perspective_correction.md), [`Camera Focus`](camera_focus.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Trace Visualization`](trace_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`JSON Parser`](json_parser.md), [`Dot Visualization`](dot_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Image Stack`](image_stack.md), [`Background Color Visualization`](background_color_visualization.md), [`Crop Visualization`](crop_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Detections Consensus`](detections_consensus.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Pixel Color Count`](pixel_color_count.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Circle Visualization`](circle_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Grid Visualization`](grid_visualization.md), [`Image Threshold`](image_threshold.md), [`Triangle Visualization`](triangle_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Local File Sink`](local_file_sink.md), [`Line Counter`](line_counter.md), [`Image Preprocessing`](image_preprocessing.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Depth Estimation`](depth_estimation.md), [`SIFT Comparison`](sift_comparison.md), [`Webhook Sink`](webhook_sink.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Template Matching`](template_matching.md), [`Mask Visualization`](mask_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stitch Images`](stitch_images.md), [`SIFT`](sift.md), [`Icon Visualization`](icon_visualization.md), [`Label Visualization`](label_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Contrast Equalization`](contrast_equalization.md), [`S3 Sink`](s3_sink.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Image Contours`](image_contours.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Blur Visualization`](blur_visualization.md)
    - outputs: [`Image Slicer`](image_slicer.md), [`OpenAI`](open_ai.md), [`LMM For Classification`](lmm_for_classification.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`SAM 3`](sam3.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Halo Visualization`](halo_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenRouter`](open_router.md), [`Color Visualization`](color_visualization.md), [`Cache Set`](cache_set.md), [`Text Display`](text_display.md), [`Line Counter Visualization`](line_counter_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Line Counter`](line_counter.md), [`Image Slicer`](image_slicer.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`SORT Tracker`](sort_tracker.md), [`Image Blur`](image_blur.md), [`Byte Tracker`](byte_tracker.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Google Gemma API`](google_gemma_api.md), [`Halo Visualization`](halo_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Perspective Correction`](perspective_correction.md), [`Trace Visualization`](trace_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Dot Visualization`](dot_visualization.md), [`Crop Visualization`](crop_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Time in Zone`](timein_zone.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Path Deviation`](path_deviation.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Pixel Color Count`](pixel_color_count.md), [`SIFT Comparison`](sift_comparison.md), [`Google Gemini`](google_gemini.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Grid Visualization`](grid_visualization.md), [`Dynamic Zone`](dynamic_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Image Preprocessing`](image_preprocessing.md), [`Detection Offset`](detection_offset.md), [`Clip Comparison`](clip_comparison.md), [`Time in Zone`](timein_zone.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Mask Visualization`](mask_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemini`](google_gemini.md), [`Label Visualization`](label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Slack Notification`](slack_notification.md), [`Size Measurement`](size_measurement.md), [`YOLO-World Model`](yolo_world_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Identify Outliers`](identify_outliers.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SAM 3`](sam3.md), [`Morphological Transformation`](morphological_transformation.md), [`Google Gemma`](google_gemma.md), [`Email Notification`](email_notification.md), [`Corner Visualization`](corner_visualization.md), [`Motion Detection`](motion_detection.md), [`VLM As Detector`](vlm_as_detector.md), [`Time in Zone`](timein_zone.md), [`Byte Tracker`](byte_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Email Notification`](email_notification.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Dominant Color`](dominant_color.md), [`Florence-2 Model`](florence2_model.md), [`Identify Changes`](identify_changes.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Seg Preview`](seg_preview.md), [`QR Code Generator`](qr_code_generator.md), [`Qwen-VL`](qwen_vl.md), [`Image Stack`](image_stack.md), [`SAM 3`](sam3.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Detections Consensus`](detections_consensus.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemini`](google_gemini.md), [`Background Subtraction`](background_subtraction.md), [`Circle Visualization`](circle_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Image Threshold`](image_threshold.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Object Detection Model`](object_detection_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Line Counter`](line_counter.md), [`Buffer`](buffer.md), [`OpenAI`](open_ai.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`SIFT Comparison`](sift_comparison.md), [`Path Deviation`](path_deviation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Webhook Sink`](webhook_sink.md), [`Object Detection Model`](object_detection_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stitch Images`](stitch_images.md), [`Icon Visualization`](icon_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Contours`](image_contours.md), [`Byte Tracker`](byte_tracker.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Blur Visualization`](blur_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md)

    
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


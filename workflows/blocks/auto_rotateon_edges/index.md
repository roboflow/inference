
# Auto Rotate on Edges



??? "Class: `AutoRotateOnEdgesBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/classical_cv/auto_rotate_on_edges/v1.py">inference.core.workflows.core_steps.classical_cv.auto_rotate_on_edges.v1.AutoRotateOnEdgesBlockV1</a>
    



Automatically deskew an image by finding the rotation angle that best aligns the
image's dominant straight edges (e.g. the sides of a document, package, panel, or
shelf) to the vertical axis, the horizontal axis, or whichever of the two is
closest, then rotating the full-resolution image by that angle with an
automatically expanded canvas so nothing is cropped.

## How This Block Works

This block estimates the rotation angle from the image's dominant gradient
orientation, then applies that rotation to the image. The block:

1. Converts the image to grayscale (color images are converted with `BGR2GRAY`)
2. Downscales a working copy so its largest dimension is at most `internal_resolution` pixels (default 1000, for speed;
   the detected angle is scale-invariant and is later applied to the full-resolution
   original)
3. Checks for a flat/low-texture image (very low pixel standard deviation) and, if
   found, skips processing and returns the image unchanged with `angle = 0.0`
4. Enhances local contrast with CLAHE and applies a light Gaussian blur to reduce
   noise sensitivity, producing a float32 image used only for scoring candidate
   angles
5. Computes Sobel gradients once and builds a magnitude-weighted histogram of
   gradient orientations (mod 180 degrees): gradients of the dominant lines
   point perpendicular to the lines themselves, so vertical lines concentrate
   the histogram around 0 degrees and horizontal lines around 90 degrees
6. Rejects indistinct histograms (no clearly dominant orientation) and returns
   the image unchanged
7. Refines the histogram's dominant mode with a doubled-angle weighted
   circular mean over nearby orientations for sub-degree precision, and
   normalizes the correction into `(-90, 90]` degrees for
   `vertical`/`horizontal` or `(-45, 45]` degrees for `either`
8. If the final angle is smaller in magnitude than `skip_below_degrees`, returns the
   image unchanged with `angle = 0.0` (avoids unnecessary re-encoding/resampling for
   already-straight images). Likewise, if the final angle is larger in magnitude
   than `max_correction_degrees`, the image is returned unchanged with
   `angle = 0.0` - a correction beyond the plausible skew range usually means the
   search aligned to a different dominant structure (e.g. a long object
   silhouette) rather than the lines of interest
9. Otherwise rotates the full-resolution original image by the final angle around
    its center, automatically expanding the output canvas so the entire rotated
    image is preserved (matching the canvas-expansion behavior of
    `roboflow_core/image_preprocessing@v1`'s rotate task)

The block outputs both the deskewed image (with expanded canvas) and the applied
rotation angle in degrees. Angles follow OpenCV's convention: positive angles
rotate counter-clockwise.

## Common Use Cases

- **Document and Label Scanning**: Straighten photographed documents, labels, or
  packaging before OCR or downstream detection, improving text and barcode
  reading accuracy
- **Fixed-Camera Inspection**: Correct small mounting-induced tilts in
  fixed-position industrial cameras so downstream measurement blocks (e.g. size or
  distance measurement) operate on axis-aligned imagery
- **Conveyor and Shelf Alignment**: Align images of conveyors, shelving, or panels
  whose edges should be vertical or horizontal, improving the reliability of
  downstream line/edge-based analysis
- **Pre-processing for Detection Models**: Straighten input images before running
  object detection or classification models that are sensitive to rotation

## Connecting to Other Blocks

This block receives an image and produces a deskewed image plus the applied angle:

- **Before detection or classification models** to straighten images prior to
  inference, enabling more reliable model outputs on tilted source imagery
- **Before measurement blocks** (e.g. distance or size measurement) that assume an
  axis-aligned scene
- **With downstream coordinate mapping** - the `angle` output (together with the
  original image's dimensions) fully determines the applied transform, so
  detections made on the rotated image can be mapped back into the original
  image's coordinate space by inverting the rotation matrix built by
  `build_auto_rotate_matrix`


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/auto_rotate_on_edges@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `target_orientation` | `str` | Which edge direction the image's dominant straight lines should be aligned to. 'vertical' searches for the rotation that makes the dominant lines vertical, 'horizontal' searches for the rotation that makes the dominant lines horizontal, and 'either' aligns to whichever of the two is closest (search is limited to a +/-45 degree range around the nearest right angle).. | ✅ |
| `skip_below_degrees` | `float` | If the detected correction angle's absolute value is smaller than this threshold (in degrees), the block returns the input image unchanged (identity passthrough) with `angle = 0.0`, avoiding unnecessary re-encoding/resampling of images that are already sufficiently straight.. | ✅ |
| `max_correction_degrees` | `float` | If the best correction angle found exceeds this cap (in absolute degrees), the block returns the input image unchanged (identity passthrough) with `angle = 0.0`. Set this when the plausible skew range of your imagery is known (e.g. parts photographed at most ~40 degrees off-axis) so that a different dominant structure in the image (e.g. a long object silhouette instead of the lines of interest) cannot cause a wild, incorrect rotation. The default of 90.0 disables the cap.. | ✅ |
| `internal_resolution` | `int` | The rotation-angle search runs on a working copy of the image downscaled so that its longest side is at most this many pixels; the full-resolution image is only used for the final rotation. Lower values make the search faster but can blur very thin lines; higher values preserve fine-line detail at the cost of search time.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Auto Rotate on Edges` in version `v1`.

    - inputs: [`Local File Sink`](local_file_sink.md), [`Text Display`](text_display.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Crop Visualization`](crop_visualization.md), [`Image Slicer`](image_slicer.md), [`Detection Event Log`](detection_event_log.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Image Threshold`](image_threshold.md), [`Image Blur`](image_blur.md), [`LMM`](lmm.md), [`Color Visualization`](color_visualization.md), [`Google Gemini`](google_gemini.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Gaze Detection`](gaze_detection.md), [`Dynamic Crop`](dynamic_crop.md), [`S3 Sink`](s3_sink.md), [`MQTT Writer`](mqtt_writer.md), [`Image Slicer`](image_slicer.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Qwen-VL`](qwen_vl.md), [`Google Gemini`](google_gemini.md), [`Template Matching`](template_matching.md), [`Current Time`](current_time.md), [`Pixel Color Count`](pixel_color_count.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Slack Notification`](slack_notification.md), [`Event Writer`](event_writer.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SIFT Comparison`](sift_comparison.md), [`SIFT`](sift.md), [`Morphological Transformation`](morphological_transformation.md), [`Background Subtraction`](background_subtraction.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemini`](google_gemini.md), [`Ellipse Visualization`](ellipse_visualization.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Trace Visualization`](trace_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Grid Visualization`](grid_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Focus`](camera_focus.md), [`Google Gemma`](google_gemma.md), [`EasyOCR`](easy_ocr.md), [`Google Gemma API`](google_gemma_api.md), [`Cosmos 3`](cosmos3.md), [`OpenAI`](open_ai.md), [`Cosine Similarity`](cosine_similarity.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Webhook Sink`](webhook_sink.md), [`Distance Measurement`](distance_measurement.md), [`VLM As Detector`](vlm_as_detector.md), [`Depth Estimation`](depth_estimation.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Blur Visualization`](blur_visualization.md), [`Image Preprocessing`](image_preprocessing.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`LMM For Classification`](lmm_for_classification.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Florence-2 Model`](florence2_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Line Counter`](line_counter.md), [`Stitch Images`](stitch_images.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`Camera Calibration`](camera_calibration.md), [`Google Vision OCR`](google_vision_ocr.md), [`OpenAI`](open_ai.md), [`Mask Visualization`](mask_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Florence-2 Model`](florence2_model.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Email Notification`](email_notification.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`VLM As Classifier`](vlm_as_classifier.md), [`SIFT Comparison`](sift_comparison.md), [`Halo Visualization`](halo_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`GLM-OCR`](glmocr.md), [`CogVLM`](cog_vlm.md), [`PLC Writer`](plc_writer.md), [`CSV Formatter`](csv_formatter.md), [`Circle Visualization`](circle_visualization.md), [`Image Contours`](image_contours.md), [`Camera Focus`](camera_focus.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Line Counter`](line_counter.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Icon Visualization`](icon_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Relative Static Crop`](relative_static_crop.md), [`Anthropic Claude`](anthropic_claude.md), [`PP-OCR`](ppocr.md), [`Label Visualization`](label_visualization.md), [`Frame Delay`](frame_delay.md), [`Background Color Visualization`](background_color_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Stack`](image_stack.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`OCR Model`](ocr_model.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Halo Visualization`](halo_visualization.md), [`OpenRouter`](open_router.md), [`Object Detection Model`](object_detection_model.md), [`Email Notification`](email_notification.md), [`Anthropic Claude`](anthropic_claude.md), [`Identify Changes`](identify_changes.md)
    - outputs: [`Crop Visualization`](crop_visualization.md), [`Image Slicer`](image_slicer.md), [`Detection Event Log`](detection_event_log.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Dynamic Crop`](dynamic_crop.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Velocity`](velocity.md), [`Image Slicer`](image_slicer.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Template Matching`](template_matching.md), [`Pixel Color Count`](pixel_color_count.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SIFT Comparison`](sift_comparison.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Morphological Transformation`](morphological_transformation.md), [`Background Subtraction`](background_subtraction.md), [`Line Counter Visualization`](line_counter_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Ellipse Visualization`](ellipse_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI`](open_ai.md), [`SAM 3`](sam3.md), [`Grid Visualization`](grid_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Google Gemma`](google_gemma.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Object Detection Model`](object_detection_model.md), [`Byte Tracker`](byte_tracker.md), [`Cosmos 3`](cosmos3.md), [`Motion Detection`](motion_detection.md), [`OpenAI`](open_ai.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Webhook Sink`](webhook_sink.md), [`VLM As Detector`](vlm_as_detector.md), [`Depth Estimation`](depth_estimation.md), [`Image Preprocessing`](image_preprocessing.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SAM 3`](sam3.md), [`QR Code Detection`](qr_code_detection.md), [`Stitch Images`](stitch_images.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Qwen3.5`](qwen3.5.md), [`Camera Calibration`](camera_calibration.md), [`Google Vision OCR`](google_vision_ocr.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`Qwen3-VL`](qwen3_vl.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Clip Comparison`](clip_comparison.md), [`Pixelate Visualization`](pixelate_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Detections Stitch`](detections_stitch.md), [`Track Class Lock`](track_class_lock.md), [`Triangle Visualization`](triangle_visualization.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Icon Visualization`](icon_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Relative Static Crop`](relative_static_crop.md), [`PP-OCR`](ppocr.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Stack`](image_stack.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`VLM As Detector`](vlm_as_detector.md), [`OCR Model`](ocr_model.md), [`SAM 3`](sam3.md), [`GeoTag Detection`](geo_tag_detection.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Halo Visualization`](halo_visualization.md), [`OpenRouter`](open_router.md), [`Frame Delay`](frame_delay.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Text Display`](text_display.md), [`Image Threshold`](image_threshold.md), [`Image Blur`](image_blur.md), [`Gaze Detection`](gaze_detection.md), [`Color Visualization`](color_visualization.md), [`LMM`](lmm.md), [`Corner Visualization`](corner_visualization.md), [`MQTT Writer`](mqtt_writer.md), [`Continue If`](continue_if.md), [`Qwen-VL`](qwen_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Google Gemini`](google_gemini.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Event Writer`](event_writer.md), [`SIFT`](sift.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Google Gemini`](google_gemini.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Trace Visualization`](trace_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Seg Preview`](seg_preview.md), [`Camera Focus`](camera_focus.md), [`EasyOCR`](easy_ocr.md), [`Google Gemma API`](google_gemma_api.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Distance Measurement`](distance_measurement.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Blur Visualization`](blur_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Florence-2 Model`](florence2_model.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Barcode Detection`](barcode_detection.md), [`Moondream2`](moondream2.md), [`Dominant Color`](dominant_color.md), [`Clip Comparison`](clip_comparison.md), [`Dynamic Zone`](dynamic_zone.md), [`Mask Visualization`](mask_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Email Notification`](email_notification.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Halo Visualization`](halo_visualization.md), [`SmolVLM2`](smol_vlm2.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`GLM-OCR`](glmocr.md), [`CogVLM`](cog_vlm.md), [`Circle Visualization`](circle_visualization.md), [`Image Contours`](image_contours.md), [`Camera Focus`](camera_focus.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`YOLO-World Model`](yolo_world_model.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Label Visualization`](label_visualization.md), [`Time in Zone`](timein_zone.md), [`Background Color Visualization`](background_color_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Buffer`](buffer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Anthropic Claude`](anthropic_claude.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Auto Rotate on Edges` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Input image (color or grayscale) to deskew. Color images are automatically converted to grayscale for angle detection; the detected rotation angle is applied to the full-resolution original image. The output includes both the rotated image (with an automatically expanded canvas) and the applied `angle` in degrees..
        - `target_orientation` (*[`string`](../kinds/string.md)*): Which edge direction the image's dominant straight lines should be aligned to. 'vertical' searches for the rotation that makes the dominant lines vertical, 'horizontal' searches for the rotation that makes the dominant lines horizontal, and 'either' aligns to whichever of the two is closest (search is limited to a +/-45 degree range around the nearest right angle)..
        - `skip_below_degrees` (*[`float`](../kinds/float.md)*): If the detected correction angle's absolute value is smaller than this threshold (in degrees), the block returns the input image unchanged (identity passthrough) with `angle = 0.0`, avoiding unnecessary re-encoding/resampling of images that are already sufficiently straight..
        - `max_correction_degrees` (*[`float`](../kinds/float.md)*): If the best correction angle found exceeds this cap (in absolute degrees), the block returns the input image unchanged (identity passthrough) with `angle = 0.0`. Set this when the plausible skew range of your imagery is known (e.g. parts photographed at most ~40 degrees off-axis) so that a different dominant structure in the image (e.g. a long object silhouette instead of the lines of interest) cannot cause a wild, incorrect rotation. The default of 90.0 disables the cap..
        - `internal_resolution` (*[`integer`](../kinds/integer.md)*): The rotation-angle search runs on a working copy of the image downscaled so that its longest side is at most this many pixels; the full-resolution image is only used for the final rotation. Lower values make the search faster but can blur very thin lines; higher values preserve fine-line detail at the cost of search time..

    - output
    
        - `image` ([`image`](../kinds/image.md)): Image in workflows.
        - `angle` ([`float`](../kinds/float.md)): Float value.



??? tip "Example JSON definition of step `Auto Rotate on Edges` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/auto_rotate_on_edges@v1",
	    "image": "$inputs.image",
	    "target_orientation": "vertical",
	    "skip_below_degrees": 0.4,
	    "max_correction_degrees": 45.0,
	    "internal_resolution": 1000
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    


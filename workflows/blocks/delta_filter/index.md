
# Delta Filter



??? "Class: `DeltaFilterBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/flow_control/delta_filter/v1.py">inference.core.workflows.core_steps.flow_control.delta_filter.v1.DeltaFilterBlockV1</a>
    



Trigger workflow execution only when an input value changes from its previous state, enabling change detection, avoiding redundant processing when values remain constant, and optimizing system efficiency by executing downstream steps only on state transitions.

## How This Block Works

This block monitors a value and only continues workflow execution when that value changes compared to its previous state. The block:

1. Takes an image (for video metadata context) and a value to monitor as input
2. Extracts video metadata from the image to identify the video stream (video_identifier)
3. Retrieves the previously cached value for this video identifier from an internal cache
4. Compares the current input value against the cached previous value
5. If the value has changed (current value ≠ previous value):
   - Updates the cache with the new value for this video identifier
   - Continues execution to the specified `next_steps` blocks, allowing downstream processing
6. If the value has not changed (current value == previous value):
   - Terminates the current workflow branch, preventing redundant downstream execution
7. Returns flow control directives that either continue to next steps or terminate the branch

The block maintains separate cached values for each video stream (identified by video_identifier), allowing it to track value changes independently across multiple video sources. This per-video tracking ensures that the filter resets appropriately when switching between different video streams. The block supports monitoring any value type (numbers, strings, detection counts, etc.), making it versatile for detecting changes in counters, metrics, detection results, or any other workflow data. By only triggering downstream blocks when values actually change, the Delta Filter prevents unnecessary processing when values remain constant, which is especially useful in video workflows where many frames may have the same detection count or metric value.

## Common Use Cases

- **Change Detection for Counters**: Trigger actions only when counter values change (e.g., execute data logging when line counter count_in changes from 5 to 6, skip processing when count remains at 6), avoiding redundant writes or updates when values are stable
- **State Transition Monitoring**: Detect transitions in system states or detection results and trigger workflows only on state changes (e.g., execute notification when detection class changes from "empty" to "occupied", skip when state remains "occupied"), preventing repeated actions for the same state
- **Conditional Data Logging**: Write to databases, CSV files, or external systems only when values change (e.g., log count changes to OPC or PLC systems, skip logging when counts are unchanged), reducing storage and network overhead
- **Event-Based Notifications**: Send alerts or notifications only when values transition (e.g., trigger email notification when zone count changes, avoid spam when count remains constant), ensuring notifications represent meaningful changes rather than repeated states
- **Optimized Processing Pipelines**: Reduce computational load in video workflows by skipping downstream processing when monitored values haven't changed (e.g., skip expensive analysis when detection count is unchanged across frames), improving overall workflow efficiency
- **Multi-Stream Change Tracking**: Monitor value changes independently across multiple video streams (e.g., track zone counts separately for different camera feeds), with automatic per-video caching ensuring correct change detection for each stream

## Connecting to Other Blocks

This block monitors values and controls workflow execution flow, and can be connected:

- **After counting or metric blocks** (e.g., Line Counter, Time in Zone, Velocity, Detection Filter) to detect when counts, metrics, or aggregated values change and conditionally trigger downstream processing based on value transitions
- **After detection blocks** (e.g., Object Detection, Classification, Keypoint Detection) to monitor detection results, class changes, or confidence metrics and execute actions only when detection outcomes change from previous frames
- **After data processing blocks** (e.g., Property Definition, Expression, Delta Filter) to track computed values or processed metrics and trigger workflows only when these computed values transition, avoiding redundant processing
- **Before data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload, Webhook Sink) to conditionally log or store data only when monitored values change, preventing duplicate entries or unnecessary writes when values remain constant
- **Before notification blocks** (e.g., Email Notification, Slack Notification, Twilio SMS Notification) to trigger alerts only when meaningful changes occur (e.g., count changes, state transitions), avoiding notification spam when values are stable
- **In video processing workflows** where per-frame values may remain constant for many frames, using the block to efficiently detect changes and trigger expensive downstream operations only when necessary, optimizing resource usage


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/delta_filter@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Delta Filter` in version `v1`.

    - inputs: [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`OCR Model`](ocr_model.md), [`Gaze Detection`](gaze_detection.md), [`Image Slicer`](image_slicer.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Distance Measurement`](distance_measurement.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Color Visualization`](color_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Data Aggregator`](data_aggregator.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Byte Tracker`](byte_tracker.md), [`Detections Consensus`](detections_consensus.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Barcode Detection`](barcode_detection.md), [`Cache Set`](cache_set.md), [`Webhook Sink`](webhook_sink.md), [`Continue If`](continue_if.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Trace Visualization`](trace_visualization.md), [`Camera Focus`](camera_focus.md), [`Object Detection Model`](object_detection_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`OpenAI`](open_ai.md), [`Buffer`](buffer.md), [`SAM 3`](sam3.md), [`Size Measurement`](size_measurement.md), [`Image Threshold`](image_threshold.md), [`Heatmap Visualization`](heatmap_visualization.md), [`SORT Tracker`](sort_tracker.md), [`Florence-2 Model`](florence2_model.md), [`Halo Visualization`](halo_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Path Deviation`](path_deviation.md), [`GLM-OCR`](glmocr.md), [`Dot Visualization`](dot_visualization.md), [`S3 Sink`](s3_sink.md), [`Path Deviation`](path_deviation.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Seg Preview`](seg_preview.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Dynamic Zone`](dynamic_zone.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Line Counter`](line_counter.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Text Display`](text_display.md), [`CSV Formatter`](csv_formatter.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Detections Merge`](detections_merge.md), [`Perspective Correction`](perspective_correction.md), [`Anthropic Claude`](anthropic_claude.md), [`Line Counter`](line_counter.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Velocity`](velocity.md), [`Depth Estimation`](depth_estimation.md), [`Overlap Filter`](overlap_filter.md), [`Rate Limiter`](rate_limiter.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Polygon Visualization`](polygon_visualization.md), [`SmolVLM2`](smol_vlm2.md), [`SIFT`](sift.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemini`](google_gemini.md), [`Label Visualization`](label_visualization.md), [`Expression`](expression.md), [`Grid Visualization`](grid_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Contrast Equalization`](contrast_equalization.md), [`Triangle Visualization`](triangle_visualization.md), [`Property Definition`](property_definition.md), [`Halo Visualization`](halo_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`Dominant Color`](dominant_color.md), [`OpenAI`](open_ai.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Email Notification`](email_notification.md), [`Slack Notification`](slack_notification.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Detections Stitch`](detections_stitch.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Object Detection Model`](object_detection_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Email Notification`](email_notification.md), [`Google Gemma API`](google_gemma_api.md), [`Google Vision OCR`](google_vision_ocr.md), [`Identify Outliers`](identify_outliers.md), [`Google Gemini`](google_gemini.md), [`Image Preprocessing`](image_preprocessing.md), [`EasyOCR`](easy_ocr.md), [`Detections Combine`](detections_combine.md), [`Object Detection Model`](object_detection_model.md), [`Cosine Similarity`](cosine_similarity.md), [`OpenAI`](open_ai.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Byte Tracker`](byte_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Time in Zone`](timein_zone.md), [`Qwen3-VL`](qwen3_vl.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`YOLO-World Model`](yolo_world_model.md), [`Inner Workflow`](inner_workflow.md), [`Detection Offset`](detection_offset.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Template Matching`](template_matching.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Stitch Images`](stitch_images.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`SIFT Comparison`](sift_comparison.md), [`Morphological Transformation`](morphological_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`CogVLM`](cog_vlm.md), [`Crop Visualization`](crop_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Florence-2 Model`](florence2_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Time in Zone`](timein_zone.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`SAM 3`](sam3.md), [`QR Code Detection`](qr_code_detection.md), [`Icon Visualization`](icon_visualization.md), [`Detections Filter`](detections_filter.md), [`Local File Sink`](local_file_sink.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Image Contours`](image_contours.md), [`JSON Parser`](json_parser.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Time in Zone`](timein_zone.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`Anthropic Claude`](anthropic_claude.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Detector`](vlm_as_detector.md), [`LMM`](lmm.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Pixel Color Count`](pixel_color_count.md), [`Identify Changes`](identify_changes.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Image Blur`](image_blur.md), [`Image Slicer`](image_slicer.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`SAM 3`](sam3.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Corner Visualization`](corner_visualization.md), [`Delta Filter`](delta_filter.md), [`Dynamic Crop`](dynamic_crop.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Keypoint Visualization`](keypoint_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Moondream2`](moondream2.md), [`QR Code Generator`](qr_code_generator.md), [`Camera Focus`](camera_focus.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Background Color Visualization`](background_color_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Cache Get`](cache_get.md)
    - outputs: None

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Delta Filter` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): not available.
        - `value` (*[`*`](../kinds/wildcard.md)*): Value to monitor for changes. Can be any data type (numbers, strings, detection counts, metrics, etc.) from workflow inputs or step outputs. The workflow branch continues to next_steps only when this value differs from the previously cached value for the current video stream. If the value remains the same, the branch terminates to avoid redundant processing. Example: Monitor a line counter count ($steps.line_counter.count_in) and trigger actions only when the count changes..
        - `next_steps` (*step*): List of workflow steps to execute when the monitored value changes from its previous state. These steps receive control flow only when a change is detected, allowing conditional downstream processing. If the value hasn't changed, these steps will not execute as the branch terminates. Each step selector references a block in the workflow that should execute on value transitions..

    - output
    




??? tip "Example JSON definition of step `Delta Filter` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/delta_filter@v1",
	    "image": "<block_does_not_provide_example>",
	    "value": "$steps.line_counter.count_in",
	    "next_steps": "$steps.write_to_csv"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    


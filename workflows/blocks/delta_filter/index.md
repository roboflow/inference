
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

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-alert-circle-outline:{ style="color: #f57c00" } `soft` — runtime `hosted_serverless`, `dedicated_deployment`; execution `remote`; input `video`
:   Block keeps per-video state in process memory (keyed by video_metadata.video_identifier). With remote step execution on stateless or multi-replica HTTP runtimes, successive requests may be served by different worker processes, so the state resets between calls and the output is meaningless for tracking / counting / aggregation. Use local step execution in an InferencePipeline for stable cross-frame results.

:material-alert-circle-outline:{ style="color: #f57c00" } `soft` — input `image`
:   Block depends on temporal context from video or repeated-frame workflows. With a still image/photo, there is no meaningful history to track, compare, aggregate, or visualize, so the block provides little or no benefit.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Delta Filter` in version `v1`.

    - inputs: [`Overlap Analysis`](overlap_analysis.md), [`Template Matching`](template_matching.md), [`Morphological Transformation`](morphological_transformation.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Blur Visualization`](blur_visualization.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Delta Filter`](delta_filter.md), [`OpenAI`](open_ai.md), [`YOLO-World Model`](yolo_world_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Anthropic Claude`](anthropic_claude.md), [`Camera Focus`](camera_focus.md), [`Track Class Lock`](track_class_lock.md), [`Size Measurement`](size_measurement.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Trace Visualization`](trace_visualization.md), [`SmolVLM2`](smol_vlm2.md), [`JSON Parser`](json_parser.md), [`Label Visualization`](label_visualization.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Florence-2 Model`](florence2_model.md), [`Text Display`](text_display.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen-VL`](qwen_vl.md), [`Image Blur`](image_blur.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Velocity`](velocity.md), [`CSV Formatter`](csv_formatter.md), [`Gaze Detection`](gaze_detection.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`LMM`](lmm.md), [`QR Code Detection`](qr_code_detection.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Camera Focus`](camera_focus.md), [`Line Counter`](line_counter.md), [`SORT Tracker`](sort_tracker.md), [`VLM As Detector`](vlm_as_detector.md), [`Qwen3-VL`](qwen3_vl.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Clip Comparison`](clip_comparison.md), [`Detections Stitch`](detections_stitch.md), [`Google Gemma API`](google_gemma_api.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Halo Visualization`](halo_visualization.md), [`Color Visualization`](color_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Event Writer`](event_writer.md), [`Buffer`](buffer.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Property Definition`](property_definition.md), [`Cache Set`](cache_set.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Time in Zone`](timein_zone.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`OpenAI`](open_ai.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Identify Outliers`](identify_outliers.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Detection Offset`](detection_offset.md), [`Dominant Color`](dominant_color.md), [`CogVLM`](cog_vlm.md), [`Detections Consensus`](detections_consensus.md), [`Object Detection Model`](object_detection_model.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Path Deviation`](path_deviation.md), [`Byte Tracker`](byte_tracker.md), [`Expression`](expression.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Detections Combine`](detections_combine.md), [`Continue If`](continue_if.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Clip Comparison`](clip_comparison.md), [`Cache Get`](cache_get.md), [`SAM 3`](sam3.md), [`OpenAI`](open_ai.md), [`SIFT Comparison`](sift_comparison.md), [`Time in Zone`](timein_zone.md), [`OCR Model`](ocr_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Slack Notification`](slack_notification.md), [`OpenRouter`](open_router.md), [`Detection Event Log`](detection_event_log.md), [`Google Vision OCR`](google_vision_ocr.md), [`Pixelate Visualization`](pixelate_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Dynamic Zone`](dynamic_zone.md), [`Google Gemma`](google_gemma.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Halo Visualization`](halo_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`GLM-OCR`](glmocr.md), [`Image Threshold`](image_threshold.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Stitch Images`](stitch_images.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Icon Visualization`](icon_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Google Gemini`](google_gemini.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Byte Tracker`](byte_tracker.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Webhook Sink`](webhook_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`QR Code Generator`](qr_code_generator.md), [`Path Deviation`](path_deviation.md), [`MQTT Writer`](mqtt_writer.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Object Detection Model`](object_detection_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Dot Visualization`](dot_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Seg Preview`](seg_preview.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Detections Merge`](detections_merge.md), [`SIFT`](sift.md), [`Google Gemini`](google_gemini.md), [`Dimension Collapse`](dimension_collapse.md), [`EasyOCR`](easy_ocr.md), [`Local File Sink`](local_file_sink.md), [`SAM 3`](sam3.md), [`Triangle Visualization`](triangle_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Time in Zone`](timein_zone.md), [`Polygon Visualization`](polygon_visualization.md), [`OpenAI`](open_ai.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Data Aggregator`](data_aggregator.md), [`Rate Limiter`](rate_limiter.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Google Gemini`](google_gemini.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`LMM For Classification`](lmm_for_classification.md), [`VLM As Detector`](vlm_as_detector.md), [`Identify Changes`](identify_changes.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Stack`](image_stack.md), [`Polygon Visualization`](polygon_visualization.md), [`Email Notification`](email_notification.md), [`Mask Visualization`](mask_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detections Filter`](detections_filter.md), [`Distance Measurement`](distance_measurement.md), [`Barcode Detection`](barcode_detection.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Keypoint Visualization`](keypoint_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Overlap Filter`](overlap_filter.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Email Notification`](email_notification.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Image Slicer`](image_slicer.md), [`Image Contours`](image_contours.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Image Preprocessing`](image_preprocessing.md), [`SAM 3`](sam3.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Depth Estimation`](depth_estimation.md), [`Pixel Color Count`](pixel_color_count.md), [`Motion Detection`](motion_detection.md), [`Current Time`](current_time.md), [`Qwen3.5`](qwen3.5.md), [`Cosine Similarity`](cosine_similarity.md), [`Corner Visualization`](corner_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Inner Workflow`](inner_workflow.md), [`Grid Visualization`](grid_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Moondream2`](moondream2.md), [`S3 Sink`](s3_sink.md), [`Circle Visualization`](circle_visualization.md), [`Image Slicer`](image_slicer.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Relative Static Crop`](relative_static_crop.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Object Detection Model`](object_detection_model.md), [`Background Color Visualization`](background_color_visualization.md), [`Line Counter`](line_counter.md)
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


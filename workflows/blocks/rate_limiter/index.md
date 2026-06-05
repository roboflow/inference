
# Rate Limiter



??? "Class: `RateLimiterBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/flow_control/rate_limiter/v1.py">inference.core.workflows.core_steps.flow_control.rate_limiter.v1.RateLimiterBlockV1</a>
    



Enforce a minimum time interval between executions of downstream workflow steps, throttling execution frequency and preventing over-execution by ensuring connected steps run no more frequently than a specified cooldown period.

## How This Block Works

This block limits the execution rate of workflow branches by enforcing a cooldown period between consecutive executions. The block:

1. Takes a cooldown period (in seconds), a `depends_on` reference, and `next_steps` as input
2. Tracks the timestamp of the last execution using an internal state variable
3. Calculates the current time:
   - For video processing: Uses video metadata (frame number and FPS) to compute a video-time-based timestamp when `video_reference_image` is provided
   - For other contexts: Uses system clock time (datetime.now())
4. Compares the time elapsed since the last execution against the `cooldown_seconds` threshold
5. If sufficient time has passed (elapsed time >= cooldown_seconds):
   - Updates the last execution timestamp
   - Continues execution to the specified `next_steps` blocks, allowing downstream processing
6. If insufficient time has passed (elapsed time < cooldown_seconds):
   - Terminates the current workflow branch, preventing downstream execution until the cooldown period expires
7. Returns flow control directives that either continue to next steps or terminate the branch

The block maintains execution state across workflow runs, tracking when downstream steps were last executed. The `depends_on` parameter establishes a dependency relationship, and the rate limiter monitors when the dependent step completes to determine if the cooldown period has elapsed. For video workflows, the block can use video-time-based timestamps (calculated from frame number and FPS) rather than wall-clock time, which is useful when processing video faster than real-time, ensuring throttling works correctly relative to video time rather than processing speed.

## Requirements

**Important Limitation**: The rate limiter currently only works in video processing contexts. When used in workflows running behind HTTP services (Roboflow Hosted API, Dedicated Deployment, or self-hosted inference server), the rate limiting will have no effect for processing HTTP requests, as each request is independent and execution state is not maintained between requests.

## Common Use Cases

- **Throttling Expensive Operations**: Limit the frequency of resource-intensive downstream operations (e.g., execute data uploads every 5 seconds maximum, skip if attempted more frequently), preventing system overload and reducing costs for operations with usage-based pricing
- **Preventing Notification Spam**: Throttle notification blocks to avoid overwhelming recipients (e.g., send email alerts at most once per minute when detections occur, skip redundant notifications), ensuring alerts remain meaningful and actionable
- **API Rate Limit Compliance**: Enforce rate limits for external API calls or service integrations (e.g., limit webhook calls to external systems to once per second, prevent exceeding API quotas), ensuring compliance with external service rate limits
- **Database Write Optimization**: Reduce write frequency to databases or data storage systems (e.g., log detection results every 10 seconds maximum, batch updates efficiently), minimizing database load and improving overall system performance
- **Video Processing Efficiency**: Control processing rate in video workflows where fast-forward processing may generate many frames quickly (e.g., throttle analysis steps to process every 2 seconds of video time, maintain proper timing when processing faster than real-time), using video-time-based throttling for accurate rate limiting
- **Resource Management**: Manage computational resources by limiting how frequently expensive model inference or processing steps execute (e.g., run expensive analysis at most once per 3 seconds, skip redundant processing), balancing processing speed with resource constraints

## Connecting to Other Blocks

This block controls workflow execution flow and can be connected:

- **Between workflow steps** where you want to throttle execution rate, placing the rate limiter between a source step (referenced in `depends_on`) and target steps (specified in `next_steps`) to enforce a minimum time interval between executions
- **Before notification blocks** (e.g., Email Notification, Slack Notification, Twilio SMS Notification) to prevent notification spam by ensuring alerts are sent no more frequently than the cooldown period, maintaining alert effectiveness and avoiding overwhelming recipients
- **Before data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload, Webhook Sink) to throttle write operations and reduce storage or network overhead, batching updates efficiently and preventing excessive write operations
- **Before external API integrations** (e.g., Webhook Sink) to comply with external service rate limits, ensuring API calls don't exceed allowed frequencies and preventing rate limit errors
- **In video processing workflows** where fast-forward processing generates frames rapidly, using `video_reference_image` to enable video-time-based throttling that works correctly even when processing video faster than real-time, maintaining proper execution timing relative to video playback time
- **After detection or analysis blocks** (e.g., Object Detection, Classification, Line Counter) to throttle downstream processing triggered by frequent detections or events, ensuring expensive operations don't execute too frequently even when detections occur on every frame


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/rate_limiter@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `cooldown_seconds` | `float` | Minimum number of seconds that must elapse between consecutive executions of the next_steps blocks. The rate limiter tracks the last execution timestamp and only allows execution to continue if at least this many seconds have passed since the previous execution. Must be greater than or equal to 0.0. For video workflows, this cooldown period is enforced based on video time (calculated from frame number and FPS) when video_reference_image is provided, rather than wall-clock time.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-alert-circle-outline:{ style="color: #f57c00" } `soft` — runtime `hosted_serverless`, `dedicated_deployment`; execution `remote`
:   Cooldown / rate-limit timer is stored in process memory. With remote step execution on stateless or multi-replica HTTP runtimes each request gets a fresh worker, so cooldown does not throttle. Cooldown only behaves as documented with local step execution inside an InferencePipeline.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Rate Limiter` in version `v1`.

    - inputs: [`Detections Classes Replacement`](detections_classes_replacement.md), [`Morphological Transformation`](morphological_transformation.md), [`Image Preprocessing`](image_preprocessing.md), [`Email Notification`](email_notification.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Halo Visualization`](halo_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Detections Transformation`](detections_transformation.md), [`Pixel Color Count`](pixel_color_count.md), [`Object Detection Model`](object_detection_model.md), [`Time in Zone`](timein_zone.md), [`Template Matching`](template_matching.md), [`Image Threshold`](image_threshold.md), [`Text Display`](text_display.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Time in Zone`](timein_zone.md), [`Qwen-VL`](qwen_vl.md), [`OpenAI`](open_ai.md), [`CogVLM`](cog_vlm.md), [`Crop Visualization`](crop_visualization.md), [`Cosine Similarity`](cosine_similarity.md), [`SAM 3`](sam3.md), [`Dot Visualization`](dot_visualization.md), [`Detections Merge`](detections_merge.md), [`Google Vision OCR`](google_vision_ocr.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Florence-2 Model`](florence2_model.md), [`Dimension Collapse`](dimension_collapse.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Polygon Visualization`](polygon_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Absolute Static Crop`](absolute_static_crop.md), [`S3 Sink`](s3_sink.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`QR Code Generator`](qr_code_generator.md), [`SIFT Comparison`](sift_comparison.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Cache Get`](cache_get.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Dynamic Zone`](dynamic_zone.md), [`Detections Filter`](detections_filter.md), [`OCR Model`](ocr_model.md), [`Byte Tracker`](byte_tracker.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Gaze Detection`](gaze_detection.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Environment Secrets Store`](environment_secrets_store.md), [`LMM For Classification`](lmm_for_classification.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Blur`](image_blur.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Object Detection Model`](object_detection_model.md), [`Blur Visualization`](blur_visualization.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Path Deviation`](path_deviation.md), [`Current Time`](current_time.md), [`SAM 3`](sam3.md), [`Perspective Correction`](perspective_correction.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Byte Tracker`](byte_tracker.md), [`Detection Offset`](detection_offset.md), [`Anthropic Claude`](anthropic_claude.md), [`MQTT Writer`](mqtt_writer.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemini`](google_gemini.md), [`Identify Changes`](identify_changes.md), [`Image Slicer`](image_slicer.md), [`SAM 3`](sam3.md), [`Depth Estimation`](depth_estimation.md), [`Detections Consensus`](detections_consensus.md), [`Detections Stitch`](detections_stitch.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Object Detection Model`](object_detection_model.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Slack Notification`](slack_notification.md), [`Overlap Analysis`](overlap_analysis.md), [`Rate Limiter`](rate_limiter.md), [`Identify Outliers`](identify_outliers.md), [`Inner Workflow`](inner_workflow.md), [`Time in Zone`](timein_zone.md), [`Image Stack`](image_stack.md), [`Delta Filter`](delta_filter.md), [`Google Gemini`](google_gemini.md), [`Cache Set`](cache_set.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Label Visualization`](label_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Size Measurement`](size_measurement.md), [`Camera Focus`](camera_focus.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`CSV Formatter`](csv_formatter.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI`](open_ai.md), [`SIFT`](sift.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Moondream2`](moondream2.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Florence-2 Model`](florence2_model.md), [`Seg Preview`](seg_preview.md), [`Overlap Filter`](overlap_filter.md), [`EasyOCR`](easy_ocr.md), [`YOLO-World Model`](yolo_world_model.md), [`Buffer`](buffer.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Local File Sink`](local_file_sink.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Icon Visualization`](icon_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Path Deviation`](path_deviation.md), [`JSON Parser`](json_parser.md), [`OpenRouter`](open_router.md), [`Dominant Color`](dominant_color.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Distance Measurement`](distance_measurement.md), [`Qwen3-VL`](qwen3_vl.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OpenAI`](open_ai.md), [`Background Color Visualization`](background_color_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Continue If`](continue_if.md), [`Byte Tracker`](byte_tracker.md), [`Google Gemini`](google_gemini.md), [`Grid Visualization`](grid_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Corner Visualization`](corner_visualization.md), [`Image Slicer`](image_slicer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Reference Path Visualization`](reference_path_visualization.md), [`SmolVLM2`](smol_vlm2.md), [`Line Counter`](line_counter.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Halo Visualization`](halo_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Webhook Sink`](webhook_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Detection Event Log`](detection_event_log.md), [`VLM As Detector`](vlm_as_detector.md), [`Relative Static Crop`](relative_static_crop.md), [`Anthropic Claude`](anthropic_claude.md), [`Expression`](expression.md), [`Clip Comparison`](clip_comparison.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SORT Tracker`](sort_tracker.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Barcode Detection`](barcode_detection.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Velocity`](velocity.md), [`Motion Detection`](motion_detection.md), [`Detections Combine`](detections_combine.md), [`Camera Calibration`](camera_calibration.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Data Aggregator`](data_aggregator.md), [`Trace Visualization`](trace_visualization.md), [`Google Gemma`](google_gemma.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Line Counter`](line_counter.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`QR Code Detection`](qr_code_detection.md), [`Circle Visualization`](circle_visualization.md), [`Email Notification`](email_notification.md), [`LMM`](lmm.md), [`Event Writer`](event_writer.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Camera Focus`](camera_focus.md), [`Contrast Equalization`](contrast_equalization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Background Subtraction`](background_subtraction.md), [`Image Contours`](image_contours.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`GLM-OCR`](glmocr.md), [`Qwen3.5`](qwen3.5.md), [`VLM As Detector`](vlm_as_detector.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Property Definition`](property_definition.md), [`Stitch Images`](stitch_images.md), [`Mask Visualization`](mask_visualization.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Polygon Visualization`](polygon_visualization.md)
    - outputs: None

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Rate Limiter` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `depends_on` (*[`*`](../kinds/wildcard.md)*): Reference to the workflow step that immediately precedes this rate limiter block. This establishes the dependency relationship - the rate limiter monitors when this step completes to determine if the cooldown period has elapsed since the last execution. The depends_on step can be any workflow block whose output triggers the rate-limited downstream processing..
        - `next_steps` (*step*): List of workflow steps to execute if the rate limit allows (i.e., sufficient time has passed since the last execution). These steps receive control flow only when the cooldown period has elapsed, enabling throttled downstream processing. If the cooldown period hasn't elapsed, these steps will not execute as the branch terminates. Each step selector references a block in the workflow that should execute when rate limiting permits..
        - `video_reference_image` (*[`image`](../kinds/image.md)*): Optional reference to a video frame image to use for video-time-based timestamp generation. When provided, the rate limiter calculates timestamps based on video metadata (frame number and FPS) rather than system clock time. This is useful when processing video faster than real-time, ensuring rate limiting works correctly relative to video playback time rather than processing speed. If not provided (None), the block uses system clock time (datetime.now()). Only applicable for video processing workflows..

    - output
    




??? tip "Example JSON definition of step `Rate Limiter` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/rate_limiter@v1",
	    "cooldown_seconds": 1.0,
	    "depends_on": "$steps.model",
	    "next_steps": [
	        "$steps.upload"
	    ],
	    "video_reference_image": "$inputs.image"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    


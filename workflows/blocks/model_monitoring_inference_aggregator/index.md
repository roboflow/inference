
# Model Monitoring Inference Aggregator



??? "Class: `ModelMonitoringInferenceAggregatorBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/sinks/roboflow/model_monitoring_inference_aggregator/v1.py">inference.core.workflows.core_steps.sinks.roboflow.model_monitoring_inference_aggregator.v1.ModelMonitoringInferenceAggregatorBlockV1</a>
    



Periodically aggregate and report a curated sample of inference predictions to Roboflow Model Monitoring by collecting predictions in memory, grouping by class, selecting the most confident prediction per class, and sending aggregated results at configurable intervals to enable efficient video processing monitoring, production analytics, and model performance tracking workflows with minimal performance overhead.

## How This Block Works

This block aggregates predictions over time and sends representative samples to Roboflow Model Monitoring at regular intervals, reducing API calls and maintaining video processing performance. The block:

1. Receives predictions and configuration:
   - Takes predictions from any supported model type (object detection, instance segmentation, keypoint detection, or classification)
   - Receives model ID for identification in Model Monitoring
   - Accepts frequency parameter specifying reporting interval in seconds
   - Receives execution mode flag (fire-and-forget)
2. Validates Roboflow API key:
   - Checks that a valid Roboflow API key is available (required for API access)
   - Raises an error if API key is missing with instructions on how to retrieve one
3. Collects predictions in memory:
   - Stores predictions in an in-memory aggregator organized by model ID
   - Accumulates predictions between reporting intervals
   - Maintains state for the duration of the workflow execution session
4. Checks reporting interval:
   - Uses cache to track last report time based on unique aggregator key
   - Calculates time elapsed since last report
   - Compares elapsed time to configured frequency threshold
   - Skips reporting if interval has not been reached (returns status message)
5. Consolidates predictions when reporting:
   - Formats all collected predictions for Model Monitoring
   - Groups predictions by class name across all collected data
   - For each class, sorts predictions by confidence (highest first)
   - Selects the most confident prediction per class as representative sample
   - Creates a curated set of predictions (one per class with highest confidence)
6. Retrieves workspace information:
   - Gets workspace ID from Roboflow API using the provided API key
   - Uses caching (15-minute expiration) to avoid repeated API calls
   - Caches workspace name using MD5 hash of API key as cache key
7. Sends aggregated data to Model Monitoring:
   - Constructs inference data payload with timestamp, source info, device ID, and server version
   - Includes system information (if available) for monitoring context
   - Sends aggregated predictions (one per class) to Roboflow Model Monitoring API
   - Flushes in-memory aggregator after sending (starts fresh collection)
   - Updates last report time in cache
8. Executes synchronously or asynchronously:
   - **Asynchronous mode (fire_and_forget=True)**: Submits task to background thread pool or FastAPI background tasks, allowing workflow to continue without waiting for API call to complete
   - **Synchronous mode (fire_and_forget=False)**: Waits for API call to complete and returns immediate status, useful for debugging and error handling
9. Returns status information:
   - Outputs error_status indicating success (False) or failure (True)
   - Outputs message with reporting status or error details
   - Provides feedback on whether aggregation was sent or skipped

The block is optimized for video processing workflows where sending every prediction would create excessive API calls and impact performance. By aggregating predictions and selecting representative samples (most confident per class), the block provides meaningful monitoring data while minimizing overhead. The interval-based reporting ensures regular updates to Model Monitoring without constant API calls.

## Common Use Cases

#### 🔍 Why Use This Block?

This block is a game-changer for projects relying on video processing in Workflows. 
With its aggregation process, it identifies the most confident predictions across classes and sends 
them at regular intervals in small messages to Roboflow backend - ensuring that video processing 
performance is impacted to the least extent.

Perfect for:

* Monitoring production line performance in real-time 🏭.

* Debugging and validating your model’s performance over time ⏱️.

* Providing actionable insights from inference workflows with minimal overhead 🔧.

#### 🚨 Limitations

* The block is should not be relied on when running Workflow in `inference` server or via HTTP request to Roboflow 
hosted platform, as the internal state is not persisted in a memory that would be accessible for all requests to
the server, causing aggregation to **only have a scope of single request**. We will solve that problem in future 
releases if proven to be serious limitation for clients.

## Connecting to Other Blocks

This block receives predictions and outputs status information:

- **After model blocks** (Object Detection Model, Instance Segmentation Model, Classification Model, Keypoint Detection Model) to aggregate and report predictions to Model Monitoring (e.g., aggregate detection results, report classification outputs, monitor model predictions), enabling model-to-monitoring workflows
- **After filtering or analytics blocks** (DetectionsFilter, ContinueIf, OverlapFilter) to aggregate filtered or analyzed results for monitoring (e.g., aggregate filtered detections, report analytics results, monitor processed predictions), enabling analysis-to-monitoring workflows
- **In video processing workflows** to efficiently monitor video analysis with minimal performance impact (e.g., aggregate video frame detections, report video processing results, monitor video analysis performance), enabling video monitoring workflows
- **After preprocessing or transformation blocks** to monitor transformed predictions (e.g., aggregate transformed detections, report processed results, monitor transformation outputs), enabling transformation-to-monitoring workflows
- **In production deployment workflows** to track model performance in production environments (e.g., monitor production inference, track deployment performance, report production metrics), enabling production monitoring workflows
- **As a sink block** to send aggregated monitoring data without blocking workflow execution (e.g., background monitoring reporting, non-blocking analytics, efficient data collection), enabling sink-to-monitoring workflows

## Requirements

This block requires a valid Roboflow API key configured in the environment or workflow configuration. The API key is required to authenticate with Roboflow API and access Model Monitoring features. Visit https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key to learn how to retrieve an API key. The block maintains in-memory state for aggregation, which means it works best for long-running workflows (like video processing with InferencePipeline). The block should not be relied upon when running workflows in inference server or via HTTP requests to Roboflow hosted platform, as the internal state is only accessible for single requests and aggregation scope is limited to single request execution. The block aggregates data for all video feeds connected to a single InferencePipeline process (cannot separate aggregations per video feed). The frequency parameter must be at least 1 second. For more information on Model Monitoring at Roboflow, see https://docs.roboflow.com/deploy/model-monitoring.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/model_monitoring_inference_aggregator@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `frequency` | `int` | Reporting frequency in seconds. Specifies how often aggregated predictions are sent to Roboflow Model Monitoring. For example, if set to 5, the block collects predictions for 5 seconds, then sends the aggregated sample (one most confident prediction per class) to Model Monitoring. Must be at least 1 second. Lower values provide more frequent updates but increase API calls. Higher values reduce API calls but provide less frequent updates. Default: 5 seconds. Works well for video processing where you want regular but not excessive reporting.. | ✅ |
| `unique_aggregator_key` | `str` | Unique key used internally to track the aggregation session and cache last report time. This key must be unique for each instance of this block in your workflow. The key is used to create cache entries that track when the last report was sent, enabling interval-based reporting. This field is automatically generated and hidden in the UI.. | ❌ |
| `fire_and_forget` | `bool` | Execution mode flag. When True (default), the block runs asynchronously in the background, allowing the workflow to continue processing without waiting for the API call to complete. This provides faster workflow execution but errors are not immediately available. When False, the block runs synchronously and waits for the API call to complete, returning immediate status and error information. Use False for debugging and error handling, True for production workflows where performance is prioritized.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-cloud-off-outline:{ style="color: #546e7a" } `requires_internet` — air-gapped / offline deployments
:   This block depends on a service that is not reachable from fully offline / air-gapped deployments.

:material-alert-circle-outline:{ style="color: #f57c00" } `soft` — runtime `hosted_serverless`, `dedicated_deployment`; execution `remote`; input `video`
:   Aggregation buffers are stored in process memory while the reporting interval is tracked in cache. With remote step execution on stateless or multi-replica HTTP runtimes, predictions may be collected by different worker processes, so reports can under-collect or flush partial aggregation windows. Use local step execution in an InferencePipeline for stable video aggregation.

:material-alert-circle-outline:{ style="color: #f57c00" } `soft` — input `image`
:   Block depends on temporal context from video or repeated-frame workflows. With a still image/photo, there is no meaningful history to track, compare, aggregate, or visualize, so the block provides little or no benefit.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Model Monitoring Inference Aggregator` in version `v1`.

    - inputs: [`SAM 3`](sam3.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Time in Zone`](timein_zone.md), [`Anthropic Claude`](anthropic_claude.md), [`Dynamic Crop`](dynamic_crop.md), [`Mask Area Measurement`](mask_area_measurement.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Path Deviation`](path_deviation.md), [`SIFT Comparison`](sift_comparison.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Email Notification`](email_notification.md), [`Frame Delay`](frame_delay.md), [`EasyOCR`](easy_ocr.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Track Class Lock`](track_class_lock.md), [`Florence-2 Model`](florence2_model.md), [`Gaze Detection`](gaze_detection.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Dynamic Zone`](dynamic_zone.md), [`YOLO-World Model`](yolo_world_model.md), [`Byte Tracker`](byte_tracker.md), [`Detections Transformation`](detections_transformation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Slack Notification`](slack_notification.md), [`Byte Tracker`](byte_tracker.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Clip Comparison`](clip_comparison.md), [`PLC Reader`](plc_reader.md), [`VLM As Detector`](vlm_as_detector.md), [`CogVLM`](cog_vlm.md), [`SORT Tracker`](sort_tracker.md), [`PP-OCR`](ppocr.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OpenAI`](open_ai.md), [`Time in Zone`](timein_zone.md), [`Detection Event Log`](detection_event_log.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Object Detection Model`](object_detection_model.md), [`SAM 3`](sam3.md), [`SAM 3`](sam3.md), [`Detections Merge`](detections_merge.md), [`Current Time`](current_time.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Florence-2 Model`](florence2_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Path Deviation`](path_deviation.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Moondream2`](moondream2.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Detection Offset`](detection_offset.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`VLM As Detector`](vlm_as_detector.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Line Counter`](line_counter.md), [`MQTT Writer`](mqtt_writer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Local File Sink`](local_file_sink.md), [`Identify Outliers`](identify_outliers.md), [`Bounding Rectangle`](bounding_rectangle.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`Event Writer`](event_writer.md), [`Seg Preview`](seg_preview.md), [`Byte Tracker`](byte_tracker.md), [`Google Gemini`](google_gemini.md), [`OpenAI`](open_ai.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Detections Filter`](detections_filter.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Velocity`](velocity.md), [`Webhook Sink`](webhook_sink.md), [`Template Matching`](template_matching.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Google Gemini`](google_gemini.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`LMM`](lmm.md), [`Detections Combine`](detections_combine.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Time in Zone`](timein_zone.md), [`OCR Model`](ocr_model.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Email Notification`](email_notification.md), [`Cosmos 3`](cosmos3.md), [`Qwen-VL`](qwen_vl.md), [`PLC Writer`](plc_writer.md), [`Google Gemma`](google_gemma.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Motion Detection`](motion_detection.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Google Gemma API`](google_gemma_api.md), [`Detections Consensus`](detections_consensus.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`CSV Formatter`](csv_formatter.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Detections Stitch`](detections_stitch.md), [`GLM-OCR`](glmocr.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Vision OCR`](google_vision_ocr.md), [`Perspective Correction`](perspective_correction.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`JSON Parser`](json_parser.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenRouter`](open_router.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Identify Changes`](identify_changes.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`OpenAI`](open_ai.md), [`SIFT Comparison`](sift_comparison.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Overlap Filter`](overlap_filter.md), [`S3 Sink`](s3_sink.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md)
    - outputs: [`SAM 3`](sam3.md), [`Image Preprocessing`](image_preprocessing.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Time in Zone`](timein_zone.md), [`Anthropic Claude`](anthropic_claude.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Dynamic Crop`](dynamic_crop.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Cache Get`](cache_get.md), [`Object Detection Model`](object_detection_model.md), [`Path Deviation`](path_deviation.md), [`SIFT Comparison`](sift_comparison.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Email Notification`](email_notification.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Distance Measurement`](distance_measurement.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Florence-2 Model`](florence2_model.md), [`Gaze Detection`](gaze_detection.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Dynamic Zone`](dynamic_zone.md), [`YOLO-World Model`](yolo_world_model.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Slack Notification`](slack_notification.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Clip Comparison`](clip_comparison.md), [`CogVLM`](cog_vlm.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Corner Visualization`](corner_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Color Visualization`](color_visualization.md), [`OpenAI`](open_ai.md), [`Time in Zone`](timein_zone.md), [`Triangle Visualization`](triangle_visualization.md), [`Object Detection Model`](object_detection_model.md), [`SAM 3`](sam3.md), [`Image Threshold`](image_threshold.md), [`SAM 3`](sam3.md), [`Current Time`](current_time.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`QR Code Generator`](qr_code_generator.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Florence-2 Model`](florence2_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Path Deviation`](path_deviation.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Moondream2`](moondream2.md), [`Label Visualization`](label_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Line Counter`](line_counter.md), [`MQTT Writer`](mqtt_writer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Local File Sink`](local_file_sink.md), [`Cache Set`](cache_set.md), [`Event Writer`](event_writer.md), [`Google Gemini`](google_gemini.md), [`Depth Estimation`](depth_estimation.md), [`Seg Preview`](seg_preview.md), [`Google Gemini`](google_gemini.md), [`OpenAI`](open_ai.md), [`Trace Visualization`](trace_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Pixel Color Count`](pixel_color_count.md), [`SAM 3 Interactive`](sam3_interactive.md), [`LMM For Classification`](lmm_for_classification.md), [`Object Detection Model`](object_detection_model.md), [`Webhook Sink`](webhook_sink.md), [`Halo Visualization`](halo_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Template Matching`](template_matching.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Pixelate Visualization`](pixelate_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dot Visualization`](dot_visualization.md), [`Image Stack`](image_stack.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Google Gemini`](google_gemini.md), [`Keypoint Visualization`](keypoint_visualization.md), [`LMM`](lmm.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Time in Zone`](timein_zone.md), [`Circle Visualization`](circle_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Email Notification`](email_notification.md), [`Halo Visualization`](halo_visualization.md), [`Cosmos 3`](cosmos3.md), [`Polygon Visualization`](polygon_visualization.md), [`Qwen-VL`](qwen_vl.md), [`PLC Writer`](plc_writer.md), [`Google Gemma`](google_gemma.md), [`Crop Visualization`](crop_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Size Measurement`](size_measurement.md), [`Icon Visualization`](icon_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Motion Detection`](motion_detection.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Google Gemma API`](google_gemma_api.md), [`Detections Consensus`](detections_consensus.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Image Blur`](image_blur.md), [`Background Color Visualization`](background_color_visualization.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Detections Stitch`](detections_stitch.md), [`Blur Visualization`](blur_visualization.md), [`GLM-OCR`](glmocr.md), [`Anthropic Claude`](anthropic_claude.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Google Vision OCR`](google_vision_ocr.md), [`Perspective Correction`](perspective_correction.md), [`Polygon Visualization`](polygon_visualization.md), [`Contrast Equalization`](contrast_equalization.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Text Display`](text_display.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenRouter`](open_router.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`OpenAI`](open_ai.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`S3 Sink`](s3_sink.md), [`Line Counter`](line_counter.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Model Monitoring Inference Aggregator` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `predictions` (*Union[[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md), [`classification_prediction`](../kinds/classification_prediction.md)]*): Model predictions (object detection, instance segmentation, keypoint detection, or classification) to aggregate and report to Roboflow Model Monitoring. Predictions are collected in memory, grouped by class name, and the most confident prediction per class is selected as a representative sample. Predictions accumulate between reporting intervals based on the frequency setting. Supported prediction types: supervision Detections objects or classification prediction dictionaries..
        - `model_id` (*[`roboflow_model_id`](../kinds/roboflow_model_id.md)*): Roboflow model ID (format: 'project/version') to associate with the predictions in Model Monitoring. This identifies which model generated the predictions being reported. The model ID is included in the monitoring data sent to Roboflow, allowing you to track performance per model in the Model Monitoring dashboard..
        - `frequency` (*[`string`](../kinds/string.md)*): Reporting frequency in seconds. Specifies how often aggregated predictions are sent to Roboflow Model Monitoring. For example, if set to 5, the block collects predictions for 5 seconds, then sends the aggregated sample (one most confident prediction per class) to Model Monitoring. Must be at least 1 second. Lower values provide more frequent updates but increase API calls. Higher values reduce API calls but provide less frequent updates. Default: 5 seconds. Works well for video processing where you want regular but not excessive reporting..
        - `fire_and_forget` (*[`boolean`](../kinds/boolean.md)*): Execution mode flag. When True (default), the block runs asynchronously in the background, allowing the workflow to continue processing without waiting for the API call to complete. This provides faster workflow execution but errors are not immediately available. When False, the block runs synchronously and waits for the API call to complete, returning immediate status and error information. Use False for debugging and error handling, True for production workflows where performance is prioritized..

    - output
    
        - `error_status` ([`boolean`](../kinds/boolean.md)): Boolean flag.
        - `message` ([`string`](../kinds/string.md)): String value.



??? tip "Example JSON definition of step `Model Monitoring Inference Aggregator` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/model_monitoring_inference_aggregator@v1",
	    "predictions": "$steps.object_detection.predictions",
	    "model_id": "my_project/3",
	    "frequency": 3,
	    "unique_aggregator_key": "session-1v73kdhfse",
	    "fire_and_forget": true
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    


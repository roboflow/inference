
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

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Model Monitoring Inference Aggregator` in version `v1`.

    - inputs: [`Object Detection Model`](object_detection_model.md), [`Perspective Correction`](perspective_correction.md), [`SAM 3`](sam3.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`S3 Sink`](s3_sink.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Clip Comparison`](clip_comparison.md), [`Path Deviation`](path_deviation.md), [`VLM As Detector`](vlm_as_detector.md), [`Object Detection Model`](object_detection_model.md), [`Line Counter`](line_counter.md), [`SAM 3`](sam3.md), [`Email Notification`](email_notification.md), [`Qwen-VL`](qwen_vl.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`OpenRouter`](open_router.md), [`YOLO-World Model`](yolo_world_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Time in Zone`](timein_zone.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`VLM As Detector`](vlm_as_detector.md), [`OpenAI`](open_ai.md), [`Dynamic Crop`](dynamic_crop.md), [`Detections Consensus`](detections_consensus.md), [`Email Notification`](email_notification.md), [`Seg Preview`](seg_preview.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Vision OCR`](google_vision_ocr.md), [`SAM 3`](sam3.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Path Deviation`](path_deviation.md), [`Overlap Filter`](overlap_filter.md), [`Local File Sink`](local_file_sink.md), [`Detection Offset`](detection_offset.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`EasyOCR`](easy_ocr.md), [`Motion Detection`](motion_detection.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Byte Tracker`](byte_tracker.md), [`Template Matching`](template_matching.md), [`Google Gemma API`](google_gemma_api.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Google Gemini`](google_gemini.md), [`Moondream2`](moondream2.md), [`Velocity`](velocity.md), [`SIFT Comparison`](sift_comparison.md), [`Detection Event Log`](detection_event_log.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Anthropic Claude`](anthropic_claude.md), [`Object Detection Model`](object_detection_model.md), [`Time in Zone`](timein_zone.md), [`OCR Model`](ocr_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`SIFT Comparison`](sift_comparison.md), [`Slack Notification`](slack_notification.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Detections Filter`](detections_filter.md), [`Detections Merge`](detections_merge.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`LMM For Classification`](lmm_for_classification.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Dynamic Zone`](dynamic_zone.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`LMM`](lmm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Detections Transformation`](detections_transformation.md), [`Time in Zone`](timein_zone.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Detections Stitch`](detections_stitch.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Gaze Detection`](gaze_detection.md), [`Byte Tracker`](byte_tracker.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Identify Changes`](identify_changes.md), [`SORT Tracker`](sort_tracker.md), [`JSON Parser`](json_parser.md), [`GLM-OCR`](glmocr.md), [`CogVLM`](cog_vlm.md), [`Detections Combine`](detections_combine.md), [`Bounding Rectangle`](bounding_rectangle.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`CSV Formatter`](csv_formatter.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Florence-2 Model`](florence2_model.md), [`Byte Tracker`](byte_tracker.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Webhook Sink`](webhook_sink.md), [`Identify Outliers`](identify_outliers.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Google Gemma`](google_gemma.md)
    - outputs: [`S3 Sink`](s3_sink.md), [`Email Notification`](email_notification.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Morphological Transformation`](morphological_transformation.md), [`SAM 3`](sam3.md), [`Path Deviation`](path_deviation.md), [`Qwen-VL`](qwen_vl.md), [`Clip Comparison`](clip_comparison.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`YOLO-World Model`](yolo_world_model.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`OpenAI`](open_ai.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Email Notification`](email_notification.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Anthropic Claude`](anthropic_claude.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Seg Preview`](seg_preview.md), [`Google Vision OCR`](google_vision_ocr.md), [`Label Visualization`](label_visualization.md), [`SAM 3`](sam3.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Path Deviation`](path_deviation.md), [`Local File Sink`](local_file_sink.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Google Gemini`](google_gemini.md), [`Motion Detection`](motion_detection.md), [`Background Color Visualization`](background_color_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Google Gemini`](google_gemini.md), [`Polygon Visualization`](polygon_visualization.md), [`Moondream2`](moondream2.md), [`SIFT Comparison`](sift_comparison.md), [`Florence-2 Model`](florence2_model.md), [`Time in Zone`](timein_zone.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`LMM For Classification`](lmm_for_classification.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Preprocessing`](image_preprocessing.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Dynamic Zone`](dynamic_zone.md), [`Corner Visualization`](corner_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Halo Visualization`](halo_visualization.md), [`Time in Zone`](timein_zone.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Blur Visualization`](blur_visualization.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Distance Measurement`](distance_measurement.md), [`Morphological Transformation`](morphological_transformation.md), [`Trace Visualization`](trace_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Gaze Detection`](gaze_detection.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Text Display`](text_display.md), [`Florence-2 Model`](florence2_model.md), [`Icon Visualization`](icon_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Perspective Correction`](perspective_correction.md), [`SAM 3`](sam3.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Object Detection Model`](object_detection_model.md), [`Line Counter`](line_counter.md), [`QR Code Generator`](qr_code_generator.md), [`OpenRouter`](open_router.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Image Threshold`](image_threshold.md), [`Anthropic Claude`](anthropic_claude.md), [`Dynamic Crop`](dynamic_crop.md), [`Detections Consensus`](detections_consensus.md), [`Size Measurement`](size_measurement.md), [`Cache Set`](cache_set.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Depth Estimation`](depth_estimation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Template Matching`](template_matching.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Image Blur`](image_blur.md), [`Anthropic Claude`](anthropic_claude.md), [`Object Detection Model`](object_detection_model.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`OpenAI`](open_ai.md), [`Slack Notification`](slack_notification.md), [`Image Stack`](image_stack.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Cache Get`](cache_get.md), [`LMM`](lmm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Color Visualization`](color_visualization.md), [`Google Gemini`](google_gemini.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Detections Stitch`](detections_stitch.md), [`Ellipse Visualization`](ellipse_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Mask Visualization`](mask_visualization.md), [`GLM-OCR`](glmocr.md), [`Crop Visualization`](crop_visualization.md), [`Circle Visualization`](circle_visualization.md), [`CogVLM`](cog_vlm.md), [`Contrast Equalization`](contrast_equalization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Webhook Sink`](webhook_sink.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemma`](google_gemma.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Model Monitoring Inference Aggregator` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `predictions` (*Union[[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`classification_prediction`](../kinds/classification_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md)]*): Model predictions (object detection, instance segmentation, keypoint detection, or classification) to aggregate and report to Roboflow Model Monitoring. Predictions are collected in memory, grouped by class name, and the most confident prediction per class is selected as a representative sample. Predictions accumulate between reporting intervals based on the frequency setting. Supported prediction types: supervision Detections objects or classification prediction dictionaries..
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



# Roboflow Custom Metadata



??? "Class: `RoboflowCustomMetadataBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/sinks/roboflow/custom_metadata/v1.py">inference.core.workflows.core_steps.sinks.roboflow.custom_metadata.v1.RoboflowCustomMetadataBlockV1</a>
    



Attach custom metadata fields to inference results in the Roboflow Model Monitoring dashboard by extracting inference IDs from predictions and adding name-value pairs that enable filtering, analysis, and organization of inference data for monitoring workflows, production analytics, and model performance tracking.

## How This Block Works

This block adds custom metadata to inference results stored in Roboflow Model Monitoring, allowing you to attach contextual information to predictions for filtering and analysis. The block:

1. Receives model predictions and metadata configuration:
   - Takes predictions from any supported model type (object detection, instance segmentation, keypoint detection, or classification)
   - Receives field name and field value for the custom metadata to attach
   - Accepts fire-and-forget flag for execution mode
2. Validates Roboflow API key:
   - Checks that a valid Roboflow API key is available (required for API access)
   - Raises an error if API key is missing with instructions on how to retrieve one
3. Extracts inference IDs from predictions:
   - For supervision Detections objects: extracts inference IDs from the data dictionary
   - For classification predictions: extracts inference ID from the prediction dictionary
   - Collects all unique inference IDs that need metadata attached
   - Handles cases where no inference IDs are found (returns error message)
4. Retrieves workspace information:
   - Gets workspace ID from Roboflow API using the provided API key
   - Uses caching (15-minute expiration) to avoid repeated API calls for workspace lookup
   - Caches workspace name using MD5 hash of API key as cache key
5. Adds custom metadata via API:
   - Calls Roboflow API to attach custom metadata field to each inference ID
   - Associates the field name and field value with the inference results
   - Metadata becomes available in the Model Monitoring dashboard for filtering and analysis
6. Executes synchronously or asynchronously:
   - **Asynchronous mode (fire_and_forget=True)**: Submits task to background thread pool or FastAPI background tasks, allowing workflow to continue without waiting for API call to complete
   - **Synchronous mode (fire_and_forget=False)**: Waits for API call to complete and returns immediate status, useful for debugging and error handling
7. Returns status information:
   - Outputs error_status indicating success (False) or failure (True)
   - Outputs message with upload status or error details
   - Provides feedback on whether metadata was successfully attached

The block enables attaching custom metadata to inference results, making it easier to filter and analyze predictions in the Model Monitoring dashboard. For example, you can attach location labels, quality scores, processing flags, or any other contextual information that helps organize and analyze your inference data.

## Common Use Cases

- **Location-Based Filtering**: Attach location metadata to inferences for geographic analysis and filtering (e.g., tag inferences with location labels like "toronto", "warehouse_a", "production_line_1"), enabling location-based monitoring workflows
- **Quality Control Tagging**: Attach quality or validation metadata to inferences for quality tracking (e.g., tag inferences as "pass", "fail", "requires_review", "approved"), enabling quality control workflows
- **Contextual Annotation**: Add contextual information to inferences for better organization and analysis (e.g., tag with camera ID, time period, batch number, operator ID, environmental conditions), enabling contextual analysis workflows
- **Classification Enhancement**: Attach custom labels or categories to inference results beyond model predictions (e.g., tag with business logic outcomes, workflow decisions, user feedback, manual corrections), enabling enhanced classification workflows
- **Production Analytics**: Track production metrics by attaching metadata that represents operational context (e.g., tag with shift information, production batch, equipment status, performance metrics), enabling production analytics workflows
- **Filtering and Segmentation**: Enable advanced filtering in Model Monitoring dashboard by attaching metadata that represents data segments (e.g., tag with customer segment, product category, use case type, deployment environment), enabling segmentation workflows

## Connecting to Other Blocks

This block receives predictions and outputs status information:

- **After model blocks** (Object Detection Model, Instance Segmentation Model, Classification Model, Keypoint Detection Model) to attach metadata to inference results (e.g., add location tags to detections, attach quality labels to classifications, tag keypoint detections with context), enabling model-to-metadata workflows
- **After filtering or analytics blocks** (DetectionsFilter, ContinueIf, OverlapFilter) to tag filtered or analyzed results with metadata (e.g., tag filtered detections with filter criteria, attach analytics results as metadata, label processed results with workflow state), enabling analysis-to-metadata workflows
- **After conditional execution blocks** (ContinueIf, Expression) to attach metadata based on workflow decisions (e.g., tag with decision outcomes, attach conditional branch labels, mark results based on conditions), enabling conditional-to-metadata workflows
- **In parallel with other sink blocks** to combine metadata tagging with other data storage operations (e.g., tag while uploading to dataset, attach metadata while logging, combine with webhook notifications), enabling parallel sink workflows
- **Before or after visualization blocks** to ensure metadata is attached before or after visualization operations (e.g., tag visualizations with context, attach metadata to visualized results), enabling visualization workflows with metadata
- **At workflow endpoints** to ensure all inference results are tagged with metadata before workflow completion (e.g., final metadata attachment, comprehensive result tagging, complete metadata coverage), enabling end-to-end metadata workflows

## Requirements

This block requires a valid Roboflow API key configured in the environment or workflow configuration. The API key is required to authenticate with Roboflow API and access Model Monitoring features. Visit https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key to learn how to retrieve an API key. The block requires predictions that contain inference IDs (predictions must have been generated by models that include inference IDs). Supported prediction types: object detection, instance segmentation, keypoint detection, and classification. The block uses workspace caching (15-minute expiration) to optimize API calls. For more information on Model Monitoring at Roboflow, see https://docs.roboflow.com/deploy/model-monitoring.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/roboflow_custom_metadata@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `field_name` | `str` | Name of the custom metadata field to create in Roboflow Model Monitoring. This becomes the field name that can be used for filtering and analysis in the Model Monitoring dashboard. Field names should be descriptive and represent the type of metadata being attached (e.g., 'location', 'quality', 'camera_id', 'batch_number'). The field name is used to organize and categorize metadata values.. | ❌ |
| `field_value` | `str` | Value to assign to the custom metadata field. This is the actual data that will be attached to inference results and can be used for filtering and analysis in the Model Monitoring dashboard. Can be a string literal or a selector that references workflow outputs. Common values: location identifiers (e.g., 'toronto', 'warehouse_a'), quality labels (e.g., 'pass', 'fail', 'review'), identifiers (e.g., camera IDs, batch numbers), or any other contextual information relevant to your use case.. | ✅ |
| `fire_and_forget` | `bool` | Execution mode flag. When True (default), the block runs asynchronously in the background, allowing the workflow to continue processing without waiting for the API call to complete. This provides faster workflow execution but errors are not immediately available. When False, the block runs synchronously and waits for the API call to complete, returning immediate status and error information. Use False for debugging and error handling, True for production workflows where performance is prioritized.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-cloud-off-outline:{ style="color: #546e7a" } `requires_internet` — air-gapped / offline deployments
:   This block depends on a service that is not reachable from fully offline / air-gapped deployments.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Roboflow Custom Metadata` in version `v1`.

    - inputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`VLM As Detector`](vlm_as_detector.md), [`Anthropic Claude`](anthropic_claude.md), [`OpenAI`](open_ai.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Qwen-VL`](qwen_vl.md), [`Detections Combine`](detections_combine.md), [`Slack Notification`](slack_notification.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Florence-2 Model`](florence2_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Template Matching`](template_matching.md), [`Track Class Lock`](track_class_lock.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Dynamic Zone`](dynamic_zone.md), [`Webhook Sink`](webhook_sink.md), [`SAM 3`](sam3.md), [`SAM 3`](sam3.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Byte Tracker`](byte_tracker.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Detection Offset`](detection_offset.md), [`Path Deviation`](path_deviation.md), [`Florence-2 Model`](florence2_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`MQTT Writer`](mqtt_writer.md), [`PLC Writer`](plc_writer.md), [`Gaze Detection`](gaze_detection.md), [`Google Gemini`](google_gemini.md), [`Google Gemma API`](google_gemma_api.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`PLC Reader`](plc_reader.md), [`Moondream2`](moondream2.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Google Gemini`](google_gemini.md), [`EasyOCR`](easy_ocr.md), [`Motion Detection`](motion_detection.md), [`Time in Zone`](timein_zone.md), [`Time in Zone`](timein_zone.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Local File Sink`](local_file_sink.md), [`Byte Tracker`](byte_tracker.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Identify Changes`](identify_changes.md), [`Email Notification`](email_notification.md), [`S3 Sink`](s3_sink.md), [`LMM`](lmm.md), [`Detections Consensus`](detections_consensus.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`SAM 3`](sam3.md), [`Time in Zone`](timein_zone.md), [`Event Writer`](event_writer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Path Deviation`](path_deviation.md), [`Google Vision OCR`](google_vision_ocr.md), [`Byte Tracker`](byte_tracker.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Email Notification`](email_notification.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`OpenRouter`](open_router.md), [`CSV Formatter`](csv_formatter.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Detections Merge`](detections_merge.md), [`OpenAI`](open_ai.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Detections Transformation`](detections_transformation.md), [`OpenAI`](open_ai.md), [`LMM For Classification`](lmm_for_classification.md), [`Anthropic Claude`](anthropic_claude.md), [`Dynamic Crop`](dynamic_crop.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Velocity`](velocity.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Object Detection Model`](object_detection_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Seg Preview`](seg_preview.md), [`Clip Comparison`](clip_comparison.md), [`Overlap Filter`](overlap_filter.md), [`Detections Filter`](detections_filter.md), [`VLM As Classifier`](vlm_as_classifier.md), [`SIFT Comparison`](sift_comparison.md), [`CogVLM`](cog_vlm.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`SORT Tracker`](sort_tracker.md), [`OpenAI`](open_ai.md), [`Object Detection Model`](object_detection_model.md), [`Identify Outliers`](identify_outliers.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`SAM 3 Interactive`](sam3_interactive.md), [`SIFT Comparison`](sift_comparison.md), [`Line Counter`](line_counter.md), [`Current Time`](current_time.md), [`Google Gemma`](google_gemma.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Detections Stitch`](detections_stitch.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`OCR Model`](ocr_model.md), [`Perspective Correction`](perspective_correction.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`JSON Parser`](json_parser.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`GLM-OCR`](glmocr.md), [`Detection Event Log`](detection_event_log.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md)
    - outputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Threshold`](image_threshold.md), [`Triangle Visualization`](triangle_visualization.md), [`OpenAI`](open_ai.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Qwen-VL`](qwen_vl.md), [`Slack Notification`](slack_notification.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Florence-2 Model`](florence2_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Template Matching`](template_matching.md), [`Dynamic Zone`](dynamic_zone.md), [`Halo Visualization`](halo_visualization.md), [`Webhook Sink`](webhook_sink.md), [`SAM 3`](sam3.md), [`SAM 3`](sam3.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`QR Code Generator`](qr_code_generator.md), [`Google Gemini`](google_gemini.md), [`YOLO-World Model`](yolo_world_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Halo Visualization`](halo_visualization.md), [`Label Visualization`](label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Path Deviation`](path_deviation.md), [`Florence-2 Model`](florence2_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Morphological Transformation`](morphological_transformation.md), [`Text Display`](text_display.md), [`MQTT Writer`](mqtt_writer.md), [`PLC Writer`](plc_writer.md), [`Gaze Detection`](gaze_detection.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Google Gemini`](google_gemini.md), [`Google Gemma API`](google_gemma_api.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Polygon Visualization`](polygon_visualization.md), [`Moondream2`](moondream2.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Google Gemini`](google_gemini.md), [`Motion Detection`](motion_detection.md), [`Time in Zone`](timein_zone.md), [`Time in Zone`](timein_zone.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Polygon Visualization`](polygon_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Image Preprocessing`](image_preprocessing.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Local File Sink`](local_file_sink.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Email Notification`](email_notification.md), [`Line Counter`](line_counter.md), [`S3 Sink`](s3_sink.md), [`LMM`](lmm.md), [`Distance Measurement`](distance_measurement.md), [`Detections Consensus`](detections_consensus.md), [`Cache Set`](cache_set.md), [`Size Measurement`](size_measurement.md), [`Trace Visualization`](trace_visualization.md), [`SAM 3`](sam3.md), [`Circle Visualization`](circle_visualization.md), [`Time in Zone`](timein_zone.md), [`Event Writer`](event_writer.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Path Deviation`](path_deviation.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Blur`](image_blur.md), [`Google Vision OCR`](google_vision_ocr.md), [`Morphological Transformation`](morphological_transformation.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Email Notification`](email_notification.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`OpenRouter`](open_router.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`OpenAI`](open_ai.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Mask Visualization`](mask_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenAI`](open_ai.md), [`LMM For Classification`](lmm_for_classification.md), [`Anthropic Claude`](anthropic_claude.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Anthropic Claude`](anthropic_claude.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Object Detection Model`](object_detection_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Cache Get`](cache_get.md), [`Seg Preview`](seg_preview.md), [`Clip Comparison`](clip_comparison.md), [`Camera Calibration`](camera_calibration.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Background Color Visualization`](background_color_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`CogVLM`](cog_vlm.md), [`Corner Visualization`](corner_visualization.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`OpenAI`](open_ai.md), [`Object Detection Model`](object_detection_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Depth Estimation`](depth_estimation.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Image Stack`](image_stack.md), [`Line Counter`](line_counter.md), [`Current Time`](current_time.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Google Gemma`](google_gemma.md), [`Line Counter Visualization`](line_counter_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Keypoint Visualization`](keypoint_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Detections Stitch`](detections_stitch.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Perspective Correction`](perspective_correction.md), [`Crop Visualization`](crop_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`GLM-OCR`](glmocr.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Pixel Color Count`](pixel_color_count.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Roboflow Custom Metadata` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `predictions` (*Union[[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md), [`classification_prediction`](../kinds/classification_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md)]*): Model predictions (object detection, instance segmentation, keypoint detection, or classification) to attach custom metadata to. The predictions must contain inference IDs that are used to associate metadata with specific inference results in Roboflow Model Monitoring. Inference IDs are automatically extracted from supervision Detections objects or classification prediction dictionaries. The metadata will be attached to all inference IDs found in the predictions..
        - `field_value` (*[`string`](../kinds/string.md)*): Value to assign to the custom metadata field. This is the actual data that will be attached to inference results and can be used for filtering and analysis in the Model Monitoring dashboard. Can be a string literal or a selector that references workflow outputs. Common values: location identifiers (e.g., 'toronto', 'warehouse_a'), quality labels (e.g., 'pass', 'fail', 'review'), identifiers (e.g., camera IDs, batch numbers), or any other contextual information relevant to your use case..
        - `fire_and_forget` (*[`boolean`](../kinds/boolean.md)*): Execution mode flag. When True (default), the block runs asynchronously in the background, allowing the workflow to continue processing without waiting for the API call to complete. This provides faster workflow execution but errors are not immediately available. When False, the block runs synchronously and waits for the API call to complete, returning immediate status and error information. Use False for debugging and error handling, True for production workflows where performance is prioritized..

    - output
    
        - `error_status` ([`boolean`](../kinds/boolean.md)): Boolean flag.
        - `message` ([`string`](../kinds/string.md)): String value.



??? tip "Example JSON definition of step `Roboflow Custom Metadata` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/roboflow_custom_metadata@v1",
	    "predictions": "$steps.object_detection.predictions",
	    "field_name": "location",
	    "field_value": "toronto",
	    "fire_and_forget": true
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    


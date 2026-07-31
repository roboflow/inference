
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

    - inputs: [`PLC Writer`](plc_writer.md), [`Track Class Lock`](track_class_lock.md), [`Byte Tracker`](byte_tracker.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Path Deviation`](path_deviation.md), [`VLM As Detector`](vlm_as_detector.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Webhook Sink`](webhook_sink.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Motion Detection`](motion_detection.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`YOLO-World Model`](yolo_world_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Slack Notification`](slack_notification.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`PLC Reader`](plc_reader.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`S3 Sink`](s3_sink.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Email Notification`](email_notification.md), [`Velocity`](velocity.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Detection Event Log`](detection_event_log.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`CSV Formatter`](csv_formatter.md), [`Path Deviation`](path_deviation.md), [`Detection Offset`](detection_offset.md), [`Qwen-VL`](qwen_vl.md), [`JSON Parser`](json_parser.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Seg Preview`](seg_preview.md), [`Google Gemini`](google_gemini.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`LMM For Classification`](lmm_for_classification.md), [`PP-OCR`](ppocr.md), [`SIFT Comparison`](sift_comparison.md), [`Object Detection Model`](object_detection_model.md), [`Local File Sink`](local_file_sink.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Dynamic Zone`](dynamic_zone.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Google Vision OCR`](google_vision_ocr.md), [`Google Gemma API`](google_gemma_api.md), [`Detections Filter`](detections_filter.md), [`Google Gemma`](google_gemma.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Detections Transformation`](detections_transformation.md), [`Identify Changes`](identify_changes.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`OpenAI`](open_ai.md), [`Object Detection Model`](object_detection_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Clip Comparison`](clip_comparison.md), [`OpenAI`](open_ai.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Moondream2`](moondream2.md), [`Nearest Neighbor Detection Match`](nearest_neighbor_detection_match.md), [`OpenAI`](open_ai.md), [`Byte Tracker`](byte_tracker.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`VLM As Detector`](vlm_as_detector.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Frame Delay`](frame_delay.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Gemini`](google_gemini.md), [`Detections Stitch`](detections_stitch.md), [`Event Writer`](event_writer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`CogVLM`](cog_vlm.md), [`Time in Zone`](timein_zone.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Email Notification`](email_notification.md), [`Current Time`](current_time.md), [`Template Matching`](template_matching.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Perspective Correction`](perspective_correction.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Overlap Filter`](overlap_filter.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Dynamic Crop`](dynamic_crop.md), [`OpenRouter`](open_router.md), [`Detections Consensus`](detections_consensus.md), [`OCR Model`](ocr_model.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Time in Zone`](timein_zone.md), [`Time in Zone`](timein_zone.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`LMM`](lmm.md), [`Detections Combine`](detections_combine.md), [`SIFT Comparison`](sift_comparison.md), [`SAM 3`](sam3.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`GLM-OCR`](glmocr.md), [`EasyOCR`](easy_ocr.md), [`Florence-2 Model`](florence2_model.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`SAM 3`](sam3.md), [`Line Counter`](line_counter.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detections Merge`](detections_merge.md), [`Google Gemini`](google_gemini.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Cosmos 3`](cosmos3.md), [`MQTT Writer`](mqtt_writer.md), [`SAM 3`](sam3.md), [`Anthropic Claude`](anthropic_claude.md), [`SORT Tracker`](sort_tracker.md), [`Gaze Detection`](gaze_detection.md), [`Anthropic Claude`](anthropic_claude.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Florence-2 Model`](florence2_model.md), [`Identify Outliers`](identify_outliers.md), [`Byte Tracker`](byte_tracker.md)
    - outputs: [`PLC Writer`](plc_writer.md), [`Image Blur`](image_blur.md), [`Path Deviation`](path_deviation.md), [`Crop Visualization`](crop_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Webhook Sink`](webhook_sink.md), [`Motion Detection`](motion_detection.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`YOLO-World Model`](yolo_world_model.md), [`Slack Notification`](slack_notification.md), [`Label Visualization`](label_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Label Visualization`](label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`S3 Sink`](s3_sink.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Email Notification`](email_notification.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Pixel Color Count`](pixel_color_count.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Path Deviation`](path_deviation.md), [`Corner Visualization`](corner_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Distance Measurement`](distance_measurement.md), [`Camera Calibration`](camera_calibration.md), [`Google Gemini`](google_gemini.md), [`Seg Preview`](seg_preview.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`LMM For Classification`](lmm_for_classification.md), [`SIFT Comparison`](sift_comparison.md), [`Trace Visualization`](trace_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Color Visualization`](color_visualization.md), [`Local File Sink`](local_file_sink.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Dynamic Zone`](dynamic_zone.md), [`Morphological Transformation`](morphological_transformation.md), [`Google Vision OCR`](google_vision_ocr.md), [`Google Gemma API`](google_gemma_api.md), [`Google Gemma`](google_gemma.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Polygon Visualization`](polygon_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Rich Label Visualization`](rich_label_visualization.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Keypoint Visualization`](keypoint_visualization.md), [`OpenAI`](open_ai.md), [`Object Detection Model`](object_detection_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`OpenAI`](open_ai.md), [`Clip Comparison`](clip_comparison.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Mask Visualization`](mask_visualization.md), [`Moondream2`](moondream2.md), [`Nearest Neighbor Detection Match`](nearest_neighbor_detection_match.md), [`OpenAI`](open_ai.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Blur Visualization`](blur_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Reference Path Visualization`](reference_path_visualization.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Gemini`](google_gemini.md), [`Detections Stitch`](detections_stitch.md), [`Event Writer`](event_writer.md), [`Cache Set`](cache_set.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`CogVLM`](cog_vlm.md), [`Time in Zone`](timein_zone.md), [`Email Notification`](email_notification.md), [`Current Time`](current_time.md), [`Template Matching`](template_matching.md), [`Perspective Correction`](perspective_correction.md), [`Halo Visualization`](halo_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Cache Get`](cache_get.md), [`OpenRouter`](open_router.md), [`Dynamic Crop`](dynamic_crop.md), [`Detections Consensus`](detections_consensus.md), [`Halo Visualization`](halo_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Time in Zone`](timein_zone.md), [`Time in Zone`](timein_zone.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Size Measurement`](size_measurement.md), [`LMM`](lmm.md), [`Depth Estimation`](depth_estimation.md), [`SAM 3`](sam3.md), [`Dot Visualization`](dot_visualization.md), [`GLM-OCR`](glmocr.md), [`Florence-2 Model`](florence2_model.md), [`Background Color Visualization`](background_color_visualization.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Contrast Equalization`](contrast_equalization.md), [`Morphological Transformation`](morphological_transformation.md), [`Image Threshold`](image_threshold.md), [`SAM 3`](sam3.md), [`Line Counter`](line_counter.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Circle Visualization`](circle_visualization.md), [`QR Code Generator`](qr_code_generator.md), [`Google Gemini`](google_gemini.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Cosmos 3`](cosmos3.md), [`Auto Rotate on Edges`](auto_rotateon_edges.md), [`MQTT Writer`](mqtt_writer.md), [`SAM 3`](sam3.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Preprocessing`](image_preprocessing.md), [`Gaze Detection`](gaze_detection.md), [`Line Counter`](line_counter.md), [`Anthropic Claude`](anthropic_claude.md), [`Text Display`](text_display.md), [`Image Stack`](image_stack.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Florence-2 Model`](florence2_model.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Pixelate Visualization`](pixelate_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Roboflow Custom Metadata` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `predictions` (*Union[[`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`classification_prediction`](../kinds/classification_prediction.md), [`object_detection_prediction`](../kinds/object_detection_prediction.md)]*): Model predictions (object detection, instance segmentation, keypoint detection, or classification) to attach custom metadata to. The predictions must contain inference IDs that are used to associate metadata with specific inference results in Roboflow Model Monitoring. Inference IDs are automatically extracted from supervision Detections objects or classification prediction dictionaries. The metadata will be attached to all inference IDs found in the predictions..
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


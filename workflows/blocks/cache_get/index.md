
# Cache Get



??? "Class: `CacheGetBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/cache/cache_get/v1.py">inference.core.workflows.core_steps.cache.cache_get.v1.CacheGetBlockV1</a>
    



Retrieve a previously stored value from an in-memory cache by key, using the image's video identifier as a namespace to enable data sharing between workflow steps, caching intermediate results, and avoiding redundant computations within the same workflow execution context.

## How This Block Works

This block retrieves values from an in-memory cache that was previously stored using the Cache Set block. The block:

1. Receives image and cache key:
   - Takes an input image to determine the cache namespace
   - Receives a cache key (string) identifying which value to retrieve
2. Determines cache namespace:
   - Extracts video identifier from the image's video metadata
   - Uses the video identifier as the cache namespace (isolates cache entries per video/stream)
   - Falls back to "default" namespace if no video identifier is present
3. Looks up cached value:
   - Accesses the in-memory cache dictionary for the determined namespace
   - Searches for the specified key in the cache
   - Returns the cached value if found, or False if the key does not exist
4. Returns retrieved value:
   - Outputs the cached value (can be any data type: strings, numbers, lists, detections, etc.)
   - Returns False if the key was not found in the cache
   - The output type matches whatever was originally stored with Cache Set

The cache is namespaced by video identifier, meaning different videos or streams have separate cache storage. This allows workflows processing multiple videos to maintain separate caches for each video. The cache is stored in memory and is cleared when the workflow execution completes or when the block is destroyed. Cache Get must be used in conjunction with Cache Set - values are stored with Cache Set and retrieved with Cache Get using the same key and namespace (determined by the same video identifier).

## Common Use Cases

- **Shared State Between Steps**: Store intermediate results in one workflow step and retrieve them in another step (e.g., store detection results for later analysis, cache classification predictions for filtering, share metadata between blocks), enabling state sharing workflows
- **Avoid Redundant Computations**: Cache expensive computation results and reuse them across multiple workflow steps (e.g., cache model predictions, store processed images, reuse transformation results), enabling computation caching workflows
- **Video Frame Context**: Maintain context across video frames by storing frame-specific data (e.g., cache previous frame detections, store frame sequence metadata, maintain tracking state), enabling frame context workflows
- **Conditional Workflow Logic**: Store decision results or flags that control workflow execution in subsequent steps (e.g., cache filtering decisions, store validation results, maintain workflow state), enabling conditional execution workflows
- **Data Aggregation**: Accumulate data across workflow steps by storing values in cache and retrieving/updating them (e.g., aggregate detection counts, accumulate statistics, build result collections), enabling data aggregation workflows
- **Temporary Storage**: Use cache as temporary storage for values that need to be accessed by multiple workflow steps without passing through the workflow graph (e.g., store cross-step data, maintain temporary state, share non-linear workflow data), enabling temporary storage workflows

## Connecting to Other Blocks

This block retrieves cached values and can be used throughout workflows:

- **After Cache Set block** to retrieve values that were previously stored (e.g., retrieve stored detections, get cached predictions, access stored metadata), enabling cache retrieval workflows
- **In workflow branches** to access shared cache values from parallel or conditional execution paths (e.g., retrieve shared state, access cached results, get common data), enabling branch coordination workflows
- **Before blocks that need cached data** to provide cached values as input (e.g., provide cached detections to analysis, use cached predictions for filtering, pass cached metadata to processing), enabling cached input workflows
- **In conditional logic workflows** to retrieve flags or decisions stored by Cache Set (e.g., get cached validation results, retrieve decision flags, access conditional state), enabling conditional logic workflows
- **With video processing workflows** to maintain frame-specific or video-specific cache namespaces (e.g., retrieve frame context, access video-specific cache, get stream-specific data), enabling video context workflows
- **Before output or sink blocks** to include cached data in final results (e.g., include cached aggregations, output cached statistics, return cached results), enabling output workflows

## Requirements

This block requires an input image (used to determine the cache namespace via video identifier) and a cache key (string) to look up the stored value. The block only works in LOCAL execution mode - it will raise a NotImplementedError if used in other execution modes. Values must be previously stored using the Cache Set block with the same key and namespace (same video identifier). The cache is stored in memory and is automatically cleared when the workflow execution completes. The cache is namespaced by video identifier, so different videos have separate cache storage. If a key is not found in the cache, the block returns False. The cached value can be any data type (strings, numbers, lists, detections, images, etc.) depending on what was originally stored.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/cache_get@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `key` | `str` | Cache key (string) identifying which value to retrieve from the cache. The key must match the key used when storing the value with the Cache Set block. If the key does not exist in the cache, the block returns False. Keys are case-sensitive and must be exact matches. Use descriptive keys to identify different cached values (e.g., 'detections', 'classification_result', 'frame_metadata').. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-shield-alert:{ style="color: #d32f2f" } `hard` — execution `remote`
:   Cache blocks only support LOCAL workflow step execution; remote step execution raises NotImplementedError.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Cache Get` in version `v1`.

    - inputs: [`Anthropic Claude`](anthropic_claude.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Florence-2 Model`](florence2_model.md), [`Slack Notification`](slack_notification.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Google Vision OCR`](google_vision_ocr.md), [`Clip Comparison`](clip_comparison.md), [`CSV Formatter`](csv_formatter.md), [`MQTT Writer`](mqtt_writer.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Anthropic Claude`](anthropic_claude.md), [`EasyOCR`](easy_ocr.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`PLC Writer`](plc_writer.md), [`LMM`](lmm.md), [`GLM-OCR`](glmocr.md), [`OCR Model`](ocr_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Google Gemma`](google_gemma.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Event Writer`](event_writer.md), [`Google Gemini`](google_gemini.md), [`OpenAI`](open_ai.md), [`Email Notification`](email_notification.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Object Detection Model`](object_detection_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`S3 Sink`](s3_sink.md), [`OpenAI`](open_ai.md), [`Anthropic Claude`](anthropic_claude.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Florence-2 Model`](florence2_model.md), [`Webhook Sink`](webhook_sink.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`LMM For Classification`](lmm_for_classification.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Email Notification`](email_notification.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Local File Sink`](local_file_sink.md), [`Google Gemma API`](google_gemma_api.md), [`OpenAI`](open_ai.md), [`Current Time`](current_time.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Qwen-VL`](qwen_vl.md), [`OpenRouter`](open_router.md), [`CogVLM`](cog_vlm.md)
    - outputs: [`YOLO-World Model`](yolo_world_model.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`SAM 3`](sam3.md), [`Dynamic Zone`](dynamic_zone.md), [`Byte Tracker`](byte_tracker.md), [`SORT Tracker`](sort_tracker.md), [`Slack Notification`](slack_notification.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`LMM`](lmm.md), [`Qwen3-VL`](qwen3_vl.md), [`Cosine Similarity`](cosine_similarity.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Image Slicer`](image_slicer.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Delta Filter`](delta_filter.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Pixelate Visualization`](pixelate_visualization.md), [`OpenAI`](open_ai.md), [`Camera Focus`](camera_focus.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Property Definition`](property_definition.md), [`Velocity`](velocity.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Identify Outliers`](identify_outliers.md), [`Corner Visualization`](corner_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`PLC Reader`](plc_reader.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`VLM As Detector`](vlm_as_detector.md), [`Background Subtraction`](background_subtraction.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Moondream2`](moondream2.md), [`Motion Detection`](motion_detection.md), [`SAM 3`](sam3.md), [`CogVLM`](cog_vlm.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Anthropic Claude`](anthropic_claude.md), [`QR Code Detection`](qr_code_detection.md), [`SIFT`](sift.md), [`Google Vision OCR`](google_vision_ocr.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`SIFT Comparison`](sift_comparison.md), [`Size Measurement`](size_measurement.md), [`CSV Formatter`](csv_formatter.md), [`SmolVLM2`](smol_vlm2.md), [`Qwen3.5`](qwen3.5.md), [`Contrast Equalization`](contrast_equalization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`EasyOCR`](easy_ocr.md), [`Google Gemini`](google_gemini.md), [`PLC Writer`](plc_writer.md), [`Template Matching`](template_matching.md), [`GLM-OCR`](glmocr.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Stitch Images`](stitch_images.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Cache Get`](cache_get.md), [`Circle Visualization`](circle_visualization.md), [`OpenAI`](open_ai.md), [`Barcode Detection`](barcode_detection.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Continue If`](continue_if.md), [`Image Stack`](image_stack.md), [`Relative Static Crop`](relative_static_crop.md), [`Gaze Detection`](gaze_detection.md), [`Path Deviation`](path_deviation.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`VLM As Detector`](vlm_as_detector.md), [`QR Code Generator`](qr_code_generator.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Dot Visualization`](dot_visualization.md), [`Mask Visualization`](mask_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Pixel Color Count`](pixel_color_count.md), [`Blur Visualization`](blur_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Overlap Analysis`](overlap_analysis.md), [`Trace Visualization`](trace_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Florence-2 Model`](florence2_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Dynamic Crop`](dynamic_crop.md), [`Rate Limiter`](rate_limiter.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Local File Sink`](local_file_sink.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Email Notification`](email_notification.md), [`Detections Transformation`](detections_transformation.md), [`Background Color Visualization`](background_color_visualization.md), [`Image Contours`](image_contours.md), [`Qwen-VL`](qwen_vl.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Florence-2 Model`](florence2_model.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Clip Comparison`](clip_comparison.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Cache Set`](cache_set.md), [`MQTT Writer`](mqtt_writer.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Detections Consensus`](detections_consensus.md), [`Time in Zone`](timein_zone.md), [`Inner Workflow`](inner_workflow.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`OCR Model`](ocr_model.md), [`JSON Parser`](json_parser.md), [`Byte Tracker`](byte_tracker.md), [`Image Threshold`](image_threshold.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Image Blur`](image_blur.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Depth Estimation`](depth_estimation.md), [`Google Gemini`](google_gemini.md), [`Email Notification`](email_notification.md), [`Crop Visualization`](crop_visualization.md), [`Time in Zone`](timein_zone.md), [`Track Class Lock`](track_class_lock.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Label Visualization`](label_visualization.md), [`Camera Calibration`](camera_calibration.md), [`OpenAI`](open_ai.md), [`Text Display`](text_display.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Buffer`](buffer.md), [`Image Slicer`](image_slicer.md), [`Distance Measurement`](distance_measurement.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Image Preprocessing`](image_preprocessing.md), [`SAM 3`](sam3.md), [`Google Gemma API`](google_gemma_api.md), [`Triangle Visualization`](triangle_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Identify Changes`](identify_changes.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Halo Visualization`](halo_visualization.md), [`Path Deviation`](path_deviation.md), [`Switch Case`](switch_case.md), [`Dimension Collapse`](dimension_collapse.md), [`OpenRouter`](open_router.md), [`Expression`](expression.md), [`Detection Event Log`](detection_event_log.md), [`Detections Combine`](detections_combine.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Byte Tracker`](byte_tracker.md), [`Line Counter`](line_counter.md), [`Overlap Filter`](overlap_filter.md), [`Grid Visualization`](grid_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Detections Stitch`](detections_stitch.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Seg Preview`](seg_preview.md), [`Google Gemma`](google_gemma.md), [`Line Counter`](line_counter.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Object Detection Model`](object_detection_model.md), [`Event Writer`](event_writer.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Detections Merge`](detections_merge.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Data Aggregator`](data_aggregator.md), [`Detection Offset`](detection_offset.md), [`Keypoint Visualization`](keypoint_visualization.md), [`S3 Sink`](s3_sink.md), [`Camera Focus`](camera_focus.md), [`Perspective Correction`](perspective_correction.md), [`Color Visualization`](color_visualization.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Object Detection Model`](object_detection_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Webhook Sink`](webhook_sink.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Dominant Color`](dominant_color.md), [`Icon Visualization`](icon_visualization.md), [`OpenAI`](open_ai.md), [`Detections Filter`](detections_filter.md), [`Current Time`](current_time.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Ellipse Visualization`](ellipse_visualization.md), [`SAM2 Video Tracker`](sam2_video_tracker.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Cache Get` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Input image used to determine the cache namespace. The block extracts the video identifier from the image's video metadata and uses it as the cache namespace. If no video identifier is present, the block uses 'default' as the namespace. The namespace isolates cache entries so different videos or streams have separate cache storage. Use the same image (with the same video identifier) for both Cache Set and Cache Get blocks to access the same cache namespace..
        - `key` (*[`string`](../kinds/string.md)*): Cache key (string) identifying which value to retrieve from the cache. The key must match the key used when storing the value with the Cache Set block. If the key does not exist in the cache, the block returns False. Keys are case-sensitive and must be exact matches. Use descriptive keys to identify different cached values (e.g., 'detections', 'classification_result', 'frame_metadata')..

    - output
    
        - `output` ([`*`](../kinds/wildcard.md)): Equivalent of any element.



??? tip "Example JSON definition of step `Cache Get` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/cache_get@v1",
	    "image": "$inputs.image",
	    "key": "my_cache_key"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    


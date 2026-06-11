
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

    - inputs: [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`Google Gemma API`](google_gemma_api.md), [`CSV Formatter`](csv_formatter.md), [`Google Gemini`](google_gemini.md), [`Current Time`](current_time.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Event Writer`](event_writer.md), [`Email Notification`](email_notification.md), [`LMM For Classification`](lmm_for_classification.md), [`Local File Sink`](local_file_sink.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`CogVLM`](cog_vlm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Florence-2 Model`](florence2_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Google Vision OCR`](google_vision_ocr.md), [`Clip Comparison`](clip_comparison.md), [`Object Detection Model`](object_detection_model.md), [`Qwen-VL`](qwen_vl.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Anthropic Claude`](anthropic_claude.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenRouter`](open_router.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Gemini`](google_gemini.md), [`OpenAI`](open_ai.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`LMM`](lmm.md), [`EasyOCR`](easy_ocr.md), [`VLM As Detector`](vlm_as_detector.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`OCR Model`](ocr_model.md), [`OpenAI`](open_ai.md), [`Email Notification`](email_notification.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Google Gemma`](google_gemma.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Slack Notification`](slack_notification.md), [`S3 Sink`](s3_sink.md), [`GLM-OCR`](glmocr.md), [`Google Gemini`](google_gemini.md), [`OpenAI`](open_ai.md), [`Webhook Sink`](webhook_sink.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`MQTT Writer`](mqtt_writer.md), [`Anthropic Claude`](anthropic_claude.md), [`Keypoint Detection Model`](keypoint_detection_model.md)
    - outputs: [`Google Gemma API`](google_gemma_api.md), [`Camera Focus`](camera_focus.md), [`Event Writer`](event_writer.md), [`Google Gemini`](google_gemini.md), [`Rate Limiter`](rate_limiter.md), [`Current Time`](current_time.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Camera Focus`](camera_focus.md), [`Inner Workflow`](inner_workflow.md), [`Image Preprocessing`](image_preprocessing.md), [`Background Subtraction`](background_subtraction.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SAM 3`](sam3.md), [`Image Contours`](image_contours.md), [`Color Visualization`](color_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`YOLO-World Model`](yolo_world_model.md), [`Seg Preview`](seg_preview.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`JSON Parser`](json_parser.md), [`Cache Set`](cache_set.md), [`Google Vision OCR`](google_vision_ocr.md), [`Relative Static Crop`](relative_static_crop.md), [`SIFT Comparison`](sift_comparison.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Object Detection Model`](object_detection_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`EasyOCR`](easy_ocr.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`VLM As Detector`](vlm_as_detector.md), [`SAM 3`](sam3.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Qwen3.5`](qwen3.5.md), [`Google Gemma`](google_gemma.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Identify Changes`](identify_changes.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Image Blur`](image_blur.md), [`MQTT Writer`](mqtt_writer.md), [`Buffer`](buffer.md), [`Anthropic Claude`](anthropic_claude.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Camera Calibration`](camera_calibration.md), [`VLM As Detector`](vlm_as_detector.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`SIFT Comparison`](sift_comparison.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Expression`](expression.md), [`Detections Stitch`](detections_stitch.md), [`LMM For Classification`](lmm_for_classification.md), [`Overlap Analysis`](overlap_analysis.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Barcode Detection`](barcode_detection.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Perspective Correction`](perspective_correction.md), [`Cache Get`](cache_get.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`SAM 3`](sam3.md), [`Time in Zone`](timein_zone.md), [`Detection Offset`](detection_offset.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Mask Edge Snap`](mask_edge_snap.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Contrast Equalization`](contrast_equalization.md), [`Clip Comparison`](clip_comparison.md), [`SORT Tracker`](sort_tracker.md), [`Qwen-VL`](qwen_vl.md), [`Image Stack`](image_stack.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Identify Outliers`](identify_outliers.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Dot Visualization`](dot_visualization.md), [`OpenAI`](open_ai.md), [`Dynamic Zone`](dynamic_zone.md), [`Continue If`](continue_if.md), [`Moondream2`](moondream2.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Size Measurement`](size_measurement.md), [`QR Code Detection`](qr_code_detection.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Dominant Color`](dominant_color.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Path Deviation`](path_deviation.md), [`Property Definition`](property_definition.md), [`Webhook Sink`](webhook_sink.md), [`SIFT`](sift.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Image Slicer`](image_slicer.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Halo Visualization`](halo_visualization.md), [`Dimension Collapse`](dimension_collapse.md), [`Email Notification`](email_notification.md), [`QR Code Generator`](qr_code_generator.md), [`Local File Sink`](local_file_sink.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Object Detection Model`](object_detection_model.md), [`SmolVLM2`](smol_vlm2.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Path Deviation`](path_deviation.md), [`Template Matching`](template_matching.md), [`Image Slicer`](image_slicer.md), [`Depth Estimation`](depth_estimation.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Text Display`](text_display.md), [`Byte Tracker`](byte_tracker.md), [`Time in Zone`](timein_zone.md), [`Object Detection Model`](object_detection_model.md), [`Distance Measurement`](distance_measurement.md), [`Icon Visualization`](icon_visualization.md), [`Motion Detection`](motion_detection.md), [`Blur Visualization`](blur_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Anthropic Claude`](anthropic_claude.md), [`Grid Visualization`](grid_visualization.md), [`Qwen3-VL`](qwen3_vl.md), [`OpenRouter`](open_router.md), [`Cosine Similarity`](cosine_similarity.md), [`Anthropic Claude`](anthropic_claude.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`LMM`](lmm.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Image Threshold`](image_threshold.md), [`Overlap Filter`](overlap_filter.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Data Aggregator`](data_aggregator.md), [`Trace Visualization`](trace_visualization.md), [`Circle Visualization`](circle_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Label Visualization`](label_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Morphological Transformation`](morphological_transformation.md), [`Morphological Transformation`](morphological_transformation.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Dynamic Crop`](dynamic_crop.md), [`Polygon Visualization`](polygon_visualization.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`Detections Transformation`](detections_transformation.md), [`CSV Formatter`](csv_formatter.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Gaze Detection`](gaze_detection.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Byte Tracker`](byte_tracker.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Absolute Static Crop`](absolute_static_crop.md), [`CogVLM`](cog_vlm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Stitch Images`](stitch_images.md), [`Florence-2 Model`](florence2_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Detections Merge`](detections_merge.md), [`Clip Comparison`](clip_comparison.md), [`Byte Tracker`](byte_tracker.md), [`Triangle Visualization`](triangle_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Detections Consensus`](detections_consensus.md), [`Background Color Visualization`](background_color_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Corner Visualization`](corner_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Delta Filter`](delta_filter.md), [`Detections Combine`](detections_combine.md), [`Line Counter Visualization`](line_counter_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Halo Visualization`](halo_visualization.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Line Counter`](line_counter.md), [`OCR Model`](ocr_model.md), [`OpenAI`](open_ai.md), [`Email Notification`](email_notification.md), [`Slack Notification`](slack_notification.md), [`S3 Sink`](s3_sink.md), [`GLM-OCR`](glmocr.md), [`Detection Event Log`](detection_event_log.md), [`OpenAI`](open_ai.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Detections Filter`](detections_filter.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Velocity`](velocity.md)

    
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


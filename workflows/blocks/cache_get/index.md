
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

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Cache Get` in version `v1`.

    - inputs: [`OCR Model`](ocr_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`OpenAI`](open_ai.md), [`S3 Sink`](s3_sink.md), [`Slack Notification`](slack_notification.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Email Notification`](email_notification.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Detector`](vlm_as_detector.md), [`Qwen-VL`](qwen_vl.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenAI`](open_ai.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`OpenRouter`](open_router.md), [`LMM For Classification`](lmm_for_classification.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`OpenAI`](open_ai.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Anthropic Claude`](anthropic_claude.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`OpenAI`](open_ai.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`LMM`](lmm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Email Notification`](email_notification.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Anthropic Claude`](anthropic_claude.md), [`Google Gemini`](google_gemini.md), [`Google Vision OCR`](google_vision_ocr.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Local File Sink`](local_file_sink.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemini`](google_gemini.md), [`EasyOCR`](easy_ocr.md), [`GLM-OCR`](glmocr.md), [`CogVLM`](cog_vlm.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemma API`](google_gemma_api.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`CSV Formatter`](csv_formatter.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Google Gemini`](google_gemini.md), [`Florence-2 Model`](florence2_model.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Webhook Sink`](webhook_sink.md), [`Anthropic Claude`](anthropic_claude.md), [`Florence-2 Model`](florence2_model.md), [`Object Detection Model`](object_detection_model.md), [`Google Gemma`](google_gemma.md)
    - outputs: [`S3 Sink`](s3_sink.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Email Notification`](email_notification.md), [`Morphological Transformation`](morphological_transformation.md), [`SAM 3`](sam3.md), [`Qwen-VL`](qwen_vl.md), [`Path Deviation`](path_deviation.md), [`Clip Comparison`](clip_comparison.md), [`VLM As Detector`](vlm_as_detector.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`YOLO-World Model`](yolo_world_model.md), [`Line Counter`](line_counter.md), [`Time in Zone`](timein_zone.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`OpenAI`](open_ai.md), [`VLM As Detector`](vlm_as_detector.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Email Notification`](email_notification.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Seg Preview`](seg_preview.md), [`Anthropic Claude`](anthropic_claude.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Google Vision OCR`](google_vision_ocr.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`SAM 3`](sam3.md), [`Label Visualization`](label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Camera Focus`](camera_focus.md), [`Path Deviation`](path_deviation.md), [`Qwen3.5`](qwen3.5.md), [`Overlap Filter`](overlap_filter.md), [`Local File Sink`](local_file_sink.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SmolVLM2`](smol_vlm2.md), [`Google Gemini`](google_gemini.md), [`Rate Limiter`](rate_limiter.md), [`Motion Detection`](motion_detection.md), [`Byte Tracker`](byte_tracker.md), [`Background Color Visualization`](background_color_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Google Gemini`](google_gemini.md), [`Polygon Visualization`](polygon_visualization.md), [`Moondream2`](moondream2.md), [`Velocity`](velocity.md), [`SIFT Comparison`](sift_comparison.md), [`Detection Event Log`](detection_event_log.md), [`Florence-2 Model`](florence2_model.md), [`Delta Filter`](delta_filter.md), [`Grid Visualization`](grid_visualization.md), [`Time in Zone`](timein_zone.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OCR Model`](ocr_model.md), [`Barcode Detection`](barcode_detection.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Detections Filter`](detections_filter.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`Detections Merge`](detections_merge.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Detections Stabilizer`](detections_stabilizer.md), [`LMM For Classification`](lmm_for_classification.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Preprocessing`](image_preprocessing.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`SIFT`](sift.md), [`Dynamic Zone`](dynamic_zone.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Corner Visualization`](corner_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Halo Visualization`](halo_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Qwen3-VL`](qwen3_vl.md), [`Time in Zone`](timein_zone.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Blur Visualization`](blur_visualization.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Property Definition`](property_definition.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Distance Measurement`](distance_measurement.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Trace Visualization`](trace_visualization.md), [`Morphological Transformation`](morphological_transformation.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Gaze Detection`](gaze_detection.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Dot Visualization`](dot_visualization.md), [`JSON Parser`](json_parser.md), [`Pixel Color Count`](pixel_color_count.md), [`QR Code Detection`](qr_code_detection.md), [`Background Subtraction`](background_subtraction.md), [`Text Display`](text_display.md), [`Detections Combine`](detections_combine.md), [`Bounding Rectangle`](bounding_rectangle.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Absolute Static Crop`](absolute_static_crop.md), [`CSV Formatter`](csv_formatter.md), [`Florence-2 Model`](florence2_model.md), [`Byte Tracker`](byte_tracker.md), [`Icon Visualization`](icon_visualization.md), [`Identify Outliers`](identify_outliers.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Object Detection Model`](object_detection_model.md), [`Perspective Correction`](perspective_correction.md), [`SAM 3`](sam3.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Object Detection Model`](object_detection_model.md), [`Line Counter`](line_counter.md), [`QR Code Generator`](qr_code_generator.md), [`OpenRouter`](open_router.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`OpenAI`](open_ai.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Image Threshold`](image_threshold.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Anthropic Claude`](anthropic_claude.md), [`Dynamic Crop`](dynamic_crop.md), [`Detections Consensus`](detections_consensus.md), [`Size Measurement`](size_measurement.md), [`Clip Comparison`](clip_comparison.md), [`Cache Set`](cache_set.md), [`Dominant Color`](dominant_color.md), [`Continue If`](continue_if.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Detection Offset`](detection_offset.md), [`Depth Estimation`](depth_estimation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Contours`](image_contours.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`EasyOCR`](easy_ocr.md), [`Relative Static Crop`](relative_static_crop.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Template Matching`](template_matching.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Image Blur`](image_blur.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Anthropic Claude`](anthropic_claude.md), [`Triangle Visualization`](triangle_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`SIFT Comparison`](sift_comparison.md), [`OpenAI`](open_ai.md), [`Slack Notification`](slack_notification.md), [`Image Stack`](image_stack.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Stitch Images`](stitch_images.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Buffer`](buffer.md), [`OpenAI`](open_ai.md), [`Image Slicer`](image_slicer.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Image Slicer`](image_slicer.md), [`Cosine Similarity`](cosine_similarity.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Cache Get`](cache_get.md), [`LMM`](lmm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Data Aggregator`](data_aggregator.md), [`Detections Transformation`](detections_transformation.md), [`Color Visualization`](color_visualization.md), [`Google Gemini`](google_gemini.md), [`Expression`](expression.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Camera Focus`](camera_focus.md), [`Camera Calibration`](camera_calibration.md), [`Detections Stitch`](detections_stitch.md), [`Byte Tracker`](byte_tracker.md), [`Ellipse Visualization`](ellipse_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Identify Changes`](identify_changes.md), [`SORT Tracker`](sort_tracker.md), [`Mask Visualization`](mask_visualization.md), [`GLM-OCR`](glmocr.md), [`Crop Visualization`](crop_visualization.md), [`Circle Visualization`](circle_visualization.md), [`CogVLM`](cog_vlm.md), [`Inner Workflow`](inner_workflow.md), [`Dimension Collapse`](dimension_collapse.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Contrast Equalization`](contrast_equalization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Webhook Sink`](webhook_sink.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Google Gemma`](google_gemma.md)

    
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


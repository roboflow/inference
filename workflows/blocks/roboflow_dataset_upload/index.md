
# Roboflow Dataset Upload



## v2

??? "Class: `RoboflowDatasetUploadBlockV2`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/sinks/roboflow/dataset_upload/v2.py">inference.core.workflows.core_steps.sinks.roboflow.dataset_upload.v2.RoboflowDatasetUploadBlockV2</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Upload images and model predictions to a Roboflow dataset for active learning, model improvement, and data collection, with configurable usage quotas, probabilistic sampling, batch organization, image compression, and optional annotation persistence.

## How This Block Works

This block uploads workflow images and predictions to your Roboflow dataset for storage, labeling, and model training. The block:

1. Takes images and optional model predictions (object detection, instance segmentation, keypoint detection, or classification) as input
2. Validates the Roboflow API key is available (required for uploading)
3. Applies probabilistic sampling based on `data_percentage` setting, randomly selecting a percentage of inputs to upload (e.g., 50% uploads half the data, 100% uploads everything)
4. Checks usage quotas (minutely, hourly, daily limits) to ensure uploads stay within configured rate limits for active learning strategies
5. Prepares images by resizing if they exceed maximum size (maintaining aspect ratio) and compressing to specified quality level
6. Generates labeling batch names based on the prefix and batch creation frequency (never, daily, weekly, or monthly), organizing uploaded data into batches
7. Optionally persists model predictions as annotations if `persist_predictions` is enabled, allowing predictions to serve as pre-labels for review and correction
8. Attaches registration tags to images for organization and filtering in the Roboflow platform
9. Registers the image (and annotations if enabled) to the specified Roboflow project via the Roboflow API
10. Executes synchronously or asynchronously based on `fire_and_forget` setting, allowing non-blocking uploads for faster workflow execution
11. Returns error status and messages indicating upload success, failure, or sampling skip

The block supports active learning workflows by implementing usage quotas that prevent excessive data collection, helping focus on collecting valuable training data within rate limits. The probabilistic sampling feature (new in v2) allows you to randomly sample a percentage of data for upload, enabling cost-effective data collection strategies where you want to collect representative samples rather than all data. Images are organized into labeling batches that can be automatically recreated on a schedule (daily, weekly, monthly), making it easier to manage and review collected data over time. The block can operate in fire-and-forget mode for asynchronous execution, allowing workflows to continue processing without waiting for uploads to complete, or synchronously for debugging and error handling.

## Version Differences (v2 vs v1)

**New Features in v2:**

- **Probabilistic Data Sampling**: Added `data_percentage` parameter (0-100%) that enables random sampling of data for upload. This allows you to upload only a percentage of workflow inputs (e.g., 25% samples one in four images), reducing storage and annotation costs while still collecting representative data. When sampling skips an upload, the block returns a message indicating the skip.

- **Improved Default Settings**: 
  - `max_image_size` default increased from (512, 512) to (1920, 1080) for higher resolution data collection
  - `compression_level` default increased from 75 to 95 for better image quality preservation

**Behavior Changes:**

- By default, `data_percentage` is set to 100, so v2 behaves identically to v1 unless sampling is explicitly configured
- The block now uses probabilistic sampling before quota checking and image preparation, allowing efficient filtering before resource-intensive operations

## Requirements

**API Key Required**: This block requires a valid Roboflow API key to upload data. The API key must be configured in your environment or workflow configuration. Visit https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key to learn how to retrieve an API key.

## Common Use Cases

- **Active Learning Data Collection**: Collect images and predictions from production environments where models struggle or are uncertain (e.g., low-confidence detections, edge cases), enabling iterative model improvement by gathering challenging examples for retraining
- **Probabilistic Data Sampling**: Use `data_percentage` to randomly sample a subset of data for upload (e.g., upload 20% of all detections, 50% of low-confidence cases), enabling cost-effective data collection strategies that reduce storage and annotation overhead while maintaining dataset diversity
- **Production Data Logging**: Continuously upload production inference data to Roboflow datasets for monitoring, analysis, and future model training, creating a growing dataset from real-world deployments
- **Pre-Labeled Data Collection**: Upload images with model predictions as pre-labels (when `persist_predictions` is enabled), accelerating annotation workflows by providing initial labels that can be reviewed and corrected rather than starting from scratch
- **Stratified Data Sampling**: Combine probabilistic sampling with rate limiting and quotas to selectively collect data based on specific criteria (e.g., sample 30% of detections that pass filters), ensuring diverse and balanced dataset collection without overwhelming storage or annotation resources
- **Batch-Based Labeling Workflows**: Organize uploaded data into batches with automatic recreation schedules (daily, weekly, monthly), making it easier to manage labeling tasks, track progress, and organize data collection efforts over time

## Connecting to Other Blocks

This block receives data from workflow steps and uploads it to Roboflow:

- **After detection or analysis blocks** (e.g., Object Detection Model, Instance Segmentation Model, Classification Model, Keypoint Detection Model) to upload images along with their predictions, enabling active learning by collecting inference data with model outputs for annotation and retraining
- **After filtering or analytics blocks** (e.g., Detections Filter, Continue If, Overlap Filter) to selectively upload only specific types of data (e.g., low-confidence detections, overlapping objects, specific classes), focusing data collection on valuable edge cases or interesting scenarios
- **After rate limiter blocks** (e.g., Rate Limiter) to throttle upload frequency and stay within usage quotas, ensuring controlled data collection that respects rate limits and prevents excessive storage usage
- **Image inputs or preprocessing blocks** to upload raw images or processed images (e.g., crops, transformed images) without predictions, enabling collection of image data for future labeling or analysis
- **Conditional workflows** using flow control blocks (e.g., Continue If) to upload data only when certain conditions are met (e.g., upload only when detection count exceeds threshold, upload only errors or failures), enabling selective data collection based on workflow state
- **Batch processing workflows** where multiple images or predictions are generated, allowing bulk upload of workflow outputs to Roboflow datasets with probabilistic sampling for organized and cost-effective data collection


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/roboflow_dataset_upload@v2`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `target_project` | `str` | Roboflow project identifier where uploaded images and annotations will be saved. Must be a valid project in your Roboflow workspace. The project name can be specified directly or referenced from workflow inputs.. | ✅ |
| `data_percentage` | `float` | Percentage of input data (0.0 to 100.0) to randomly sample for upload. This enables probabilistic data collection where only a subset of inputs are uploaded, reducing storage and annotation costs. For example, 25.0 uploads approximately 25% of images (one in four on average), 50.0 uploads half, and 100.0 uploads everything (no sampling). Random sampling occurs before quota checking and image processing, making it efficient for large-scale data collection workflows.. | ✅ |
| `minutely_usage_limit` | `int` | Maximum number of image uploads allowed per minute for this quota. Part of the usage quota system that enforces rate limits for active learning data collection. Uploads exceeding this limit are skipped to prevent excessive data collection. Works together with hourly_usage_limit and daily_usage_limit to provide multi-level rate limiting. Note: This quota is checked after probabilistic sampling via data_percentage.. | ❌ |
| `hourly_usage_limit` | `int` | Maximum number of image uploads allowed per hour for this quota. Part of the usage quota system that enforces rate limits for active learning data collection. Uploads exceeding this limit are skipped to prevent excessive data collection. Works together with minutely_usage_limit and daily_usage_limit to provide multi-level rate limiting. Note: This quota is checked after probabilistic sampling via data_percentage.. | ❌ |
| `daily_usage_limit` | `int` | Maximum number of image uploads allowed per day for this quota. Part of the usage quota system that enforces rate limits for active learning data collection. Uploads exceeding this limit are skipped to prevent excessive data collection. Works together with minutely_usage_limit and hourly_usage_limit to provide multi-level rate limiting. Note: This quota is checked after probabilistic sampling via data_percentage.. | ❌ |
| `usage_quota_name` | `str` | Unique identifier for tracking usage quotas (minutely, hourly, daily limits). Used internally to manage rate limiting across multiple upload operations. Each unique quota name maintains separate counters, allowing different upload strategies or data collection workflows to have independent rate limits.. | ❌ |
| `max_image_size` | `Tuple[int, int]` | Maximum dimensions (width, height) for uploaded images. Images exceeding these dimensions are automatically resized while preserving aspect ratio before uploading. Default is (1920, 1080) for higher resolution data collection. Use smaller sizes (e.g., (512, 512)) for efficient storage and faster uploads, or keep the default for preserving image quality.. | ❌ |
| `compression_level` | `int` | JPEG compression quality level for uploaded images, ranging from 1 (highest compression, smallest file size, lower quality) to 100 (no compression, largest file size, highest quality). Default is 95 for better image quality preservation. Higher values preserve more image quality but increase storage and bandwidth usage. Typical values range from 70-95 for balanced quality and size.. | ❌ |
| `registration_tags` | `List[str]` | List of tags to attach to uploaded images for organization and filtering in Roboflow. Tags can be static strings (e.g., 'location-florida', 'camera-1') or dynamic values from workflow inputs. Tags help organize collected data, filter images in Roboflow, and add metadata for dataset management. Can be an empty list if no tags are needed.. | ✅ |
| `persist_predictions` | `bool` | If True, model predictions are saved as annotations (pre-labels) in the Roboflow dataset alongside images. This enables predictions to serve as starting points for annotation, allowing reviewers to correct or approve labels rather than creating them from scratch. If False, only images are uploaded without annotations. Enabling this accelerates annotation workflows by providing initial labels.. | ✅ |
| `disable_sink` | `bool` | If True, the block execution is disabled and no uploads occur. This allows temporarily disabling data collection without removing the block from workflows, useful for testing, debugging, or conditional data collection. When disabled, returns a message indicating the sink was disabled. Default is False (uploads enabled).. | ✅ |
| `fire_and_forget` | `bool` | If True, uploads execute asynchronously (fire-and-forget mode), allowing the workflow to continue immediately without waiting for upload completion. This improves workflow performance but prevents error handling. If False, uploads execute synchronously, blocking workflow execution until completion and allowing proper error handling and status reporting. Use async mode (True) for production workflows where speed is prioritized, and sync mode (False) for debugging or when error handling is critical.. | ✅ |
| `labeling_batch_prefix` | `str` | Prefix used to generate labeling batch names for organizing uploaded images in Roboflow. Combined with the batch recreation frequency and timestamps to create batch names like 'workflows_data_collector_2024_01_15'. Batches help organize collected data for labeling, making it easier to manage and review uploaded images in groups. Can be customized to match your organization scheme.. | ✅ |
| `labeling_batches_recreation_frequency` | `str` | Frequency at which new labeling batches are automatically created for uploaded images. Options: 'never' (all images go to the same batch), 'daily' (new batch each day), 'weekly' (new batch each week), 'monthly' (new batch each month). Batch timestamps are appended to the labeling_batch_prefix to create unique batch names. Automatically organizing uploads into time-based batches simplifies dataset management and makes it easier to track and review collected data over time.. | ❌ |
| `image_name` | `str` | Optional custom name for the uploaded image. This is useful when you want to preserve the original filename or use a meaningful identifier (e.g., serial number, timestamp) for the image in the Roboflow dataset. The name should not include file extension. If not provided, a UUID will be generated automatically.. | ✅ |
| `metadata` | `Dict[str, Union[bool, float, int, str]]` | Optional key-value metadata to attach to uploaded images. Metadata is stored as user_metadata on the image in Roboflow and can be used for filtering and organization. Values can be static strings, numbers, booleans, or references to workflow inputs/steps.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-cloud-off-outline:{ style="color: #546e7a" } `requires_internet` — air-gapped / offline deployments
:   This block depends on a service that is not reachable from fully offline / air-gapped deployments.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Roboflow Dataset Upload` in version `v2`.

    - inputs: [`YOLO-World Model`](yolo_world_model.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Byte Tracker`](byte_tracker.md), [`SORT Tracker`](sort_tracker.md), [`Slack Notification`](slack_notification.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`LMM`](lmm.md), [`Qwen3-VL`](qwen3_vl.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Slicer`](image_slicer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Anthropic Claude`](anthropic_claude.md), [`Velocity`](velocity.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Identify Outliers`](identify_outliers.md), [`Morphological Transformation`](morphological_transformation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`VLM As Detector`](vlm_as_detector.md), [`Background Subtraction`](background_subtraction.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`SAM 3`](sam3.md), [`CogVLM`](cog_vlm.md), [`Anthropic Claude`](anthropic_claude.md), [`QR Code Detection`](qr_code_detection.md), [`SIFT`](sift.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Vision OCR`](google_vision_ocr.md), [`SIFT Comparison`](sift_comparison.md), [`Size Measurement`](size_measurement.md), [`Google Gemini`](google_gemini.md), [`PLC Writer`](plc_writer.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Stitch Images`](stitch_images.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Image Stack`](image_stack.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`QR Code Generator`](qr_code_generator.md), [`Dot Visualization`](dot_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Pixel Color Count`](pixel_color_count.md), [`Blur Visualization`](blur_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Dynamic Crop`](dynamic_crop.md), [`Halo Visualization`](halo_visualization.md), [`Local File Sink`](local_file_sink.md), [`Rate Limiter`](rate_limiter.md), [`Email Notification`](email_notification.md), [`Detections Transformation`](detections_transformation.md), [`Environment Secrets Store`](environment_secrets_store.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Florence-2 Model`](florence2_model.md), [`Cache Set`](cache_set.md), [`MQTT Writer`](mqtt_writer.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Time in Zone`](timein_zone.md), [`Time in Zone`](timein_zone.md), [`OCR Model`](ocr_model.md), [`Byte Tracker`](byte_tracker.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Depth Estimation`](depth_estimation.md), [`Google Gemini`](google_gemini.md), [`Time in Zone`](timein_zone.md), [`Track Class Lock`](track_class_lock.md), [`Camera Calibration`](camera_calibration.md), [`Text Display`](text_display.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Image Slicer`](image_slicer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemma API`](google_gemma_api.md), [`SAM 3`](sam3.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Path Deviation`](path_deviation.md), [`Dimension Collapse`](dimension_collapse.md), [`OpenRouter`](open_router.md), [`Detection Event Log`](detection_event_log.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Line Counter`](line_counter.md), [`Clip Comparison`](clip_comparison.md), [`Detections Stitch`](detections_stitch.md), [`Seg Preview`](seg_preview.md), [`Line Counter`](line_counter.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Object Detection Model`](object_detection_model.md), [`Event Writer`](event_writer.md), [`Data Aggregator`](data_aggregator.md), [`Detection Offset`](detection_offset.md), [`Keypoint Visualization`](keypoint_visualization.md), [`S3 Sink`](s3_sink.md), [`Camera Focus`](camera_focus.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Current Time`](current_time.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`SAM 3`](sam3.md), [`Dynamic Zone`](dynamic_zone.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Cosine Similarity`](cosine_similarity.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Delta Filter`](delta_filter.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Property Definition`](property_definition.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Corner Visualization`](corner_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`LMM For Classification`](lmm_for_classification.md), [`PLC Reader`](plc_reader.md), [`Moondream2`](moondream2.md), [`Motion Detection`](motion_detection.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`CSV Formatter`](csv_formatter.md), [`SmolVLM2`](smol_vlm2.md), [`Contrast Equalization`](contrast_equalization.md), [`Qwen3.5`](qwen3.5.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`EasyOCR`](easy_ocr.md), [`Template Matching`](template_matching.md), [`GLM-OCR`](glmocr.md), [`Cache Get`](cache_get.md), [`Circle Visualization`](circle_visualization.md), [`OpenAI`](open_ai.md), [`Barcode Detection`](barcode_detection.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Continue If`](continue_if.md), [`Relative Static Crop`](relative_static_crop.md), [`Gaze Detection`](gaze_detection.md), [`Path Deviation`](path_deviation.md), [`VLM As Detector`](vlm_as_detector.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Perspective Correction`](perspective_correction.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Trace Visualization`](trace_visualization.md), [`Overlap Analysis`](overlap_analysis.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Dominant Color`](dominant_color.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Florence-2 Model`](florence2_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Background Color Visualization`](background_color_visualization.md), [`Image Contours`](image_contours.md), [`Qwen-VL`](qwen_vl.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Clip Comparison`](clip_comparison.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Detections Consensus`](detections_consensus.md), [`Inner Workflow`](inner_workflow.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`JSON Parser`](json_parser.md), [`Image Threshold`](image_threshold.md), [`Image Blur`](image_blur.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Email Notification`](email_notification.md), [`Crop Visualization`](crop_visualization.md), [`Label Visualization`](label_visualization.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI`](open_ai.md), [`Buffer`](buffer.md), [`Distance Measurement`](distance_measurement.md), [`Image Preprocessing`](image_preprocessing.md), [`Triangle Visualization`](triangle_visualization.md), [`Identify Changes`](identify_changes.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Halo Visualization`](halo_visualization.md), [`Switch Case`](switch_case.md), [`Expression`](expression.md), [`Detections Combine`](detections_combine.md), [`Byte Tracker`](byte_tracker.md), [`Overlap Filter`](overlap_filter.md), [`Grid Visualization`](grid_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Google Gemma`](google_gemma.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Detections Merge`](detections_merge.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Color Visualization`](color_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Webhook Sink`](webhook_sink.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Icon Visualization`](icon_visualization.md), [`Detections Filter`](detections_filter.md), [`OpenAI`](open_ai.md), [`SAM2 Video Tracker`](sam2_video_tracker.md)
    - outputs: [`YOLO-World Model`](yolo_world_model.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`SAM 3`](sam3.md), [`Dynamic Zone`](dynamic_zone.md), [`Slack Notification`](slack_notification.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`LMM`](lmm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Pixelate Visualization`](pixelate_visualization.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Corner Visualization`](corner_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Moondream2`](moondream2.md), [`Motion Detection`](motion_detection.md), [`SAM 3`](sam3.md), [`CogVLM`](cog_vlm.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Vision OCR`](google_vision_ocr.md), [`SIFT Comparison`](sift_comparison.md), [`Size Measurement`](size_measurement.md), [`Contrast Equalization`](contrast_equalization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`PLC Writer`](plc_writer.md), [`Template Matching`](template_matching.md), [`Google Gemini`](google_gemini.md), [`GLM-OCR`](glmocr.md), [`Cache Get`](cache_get.md), [`Circle Visualization`](circle_visualization.md), [`OpenAI`](open_ai.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Image Stack`](image_stack.md), [`Gaze Detection`](gaze_detection.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Path Deviation`](path_deviation.md), [`QR Code Generator`](qr_code_generator.md), [`Dot Visualization`](dot_visualization.md), [`Mask Visualization`](mask_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Pixel Color Count`](pixel_color_count.md), [`Blur Visualization`](blur_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Trace Visualization`](trace_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Halo Visualization`](halo_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Florence-2 Model`](florence2_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Local File Sink`](local_file_sink.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Email Notification`](email_notification.md), [`Background Color Visualization`](background_color_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Florence-2 Model`](florence2_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Cache Set`](cache_set.md), [`Clip Comparison`](clip_comparison.md), [`MQTT Writer`](mqtt_writer.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Detections Consensus`](detections_consensus.md), [`Time in Zone`](timein_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`Image Threshold`](image_threshold.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Image Blur`](image_blur.md), [`Depth Estimation`](depth_estimation.md), [`Google Gemini`](google_gemini.md), [`Email Notification`](email_notification.md), [`Crop Visualization`](crop_visualization.md), [`Time in Zone`](timein_zone.md), [`Label Visualization`](label_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI`](open_ai.md), [`Text Display`](text_display.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemma API`](google_gemma_api.md), [`Triangle Visualization`](triangle_visualization.md), [`SAM 3`](sam3.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Distance Measurement`](distance_measurement.md), [`Image Preprocessing`](image_preprocessing.md), [`Halo Visualization`](halo_visualization.md), [`Path Deviation`](path_deviation.md), [`OpenRouter`](open_router.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Line Counter`](line_counter.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Seg Preview`](seg_preview.md), [`Google Gemma`](google_gemma.md), [`Line Counter`](line_counter.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Object Detection Model`](object_detection_model.md), [`Event Writer`](event_writer.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Keypoint Visualization`](keypoint_visualization.md), [`S3 Sink`](s3_sink.md), [`Color Visualization`](color_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Webhook Sink`](webhook_sink.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Icon Visualization`](icon_visualization.md), [`OpenAI`](open_ai.md), [`Current Time`](current_time.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Ellipse Visualization`](ellipse_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Roboflow Dataset Upload` in version `v2`  has.

???+ tip "Bindings"

    - input
    
        - `images` (*[`image`](../kinds/image.md)*): Image(s) to upload to the Roboflow dataset. Can be a single image or batch of images from workflow inputs or processing steps. Images are randomly sampled based on data_percentage, resized if they exceed max_image_size, and compressed before uploading. Supports batch processing..
        - `target_project` (*[`roboflow_project`](../kinds/roboflow_project.md)*): Roboflow project identifier where uploaded images and annotations will be saved. Must be a valid project in your Roboflow workspace. The project name can be specified directly or referenced from workflow inputs..
        - `predictions` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`classification_prediction`](../kinds/classification_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Optional model predictions to upload alongside images. Predictions are saved as annotations (pre-labels) in the Roboflow dataset when persist_predictions is enabled, allowing predictions to serve as starting points for annotation review and correction. Supports object detection, instance segmentation, keypoint detection, and classification predictions. If None, only images are uploaded..
        - `data_percentage` (*[`float`](../kinds/float.md)*): Percentage of input data (0.0 to 100.0) to randomly sample for upload. This enables probabilistic data collection where only a subset of inputs are uploaded, reducing storage and annotation costs. For example, 25.0 uploads approximately 25% of images (one in four on average), 50.0 uploads half, and 100.0 uploads everything (no sampling). Random sampling occurs before quota checking and image processing, making it efficient for large-scale data collection workflows..
        - `registration_tags` (*Union[[`string`](../kinds/string.md), [`list_of_values`](../kinds/list_of_values.md)]*): List of tags to attach to uploaded images for organization and filtering in Roboflow. Tags can be static strings (e.g., 'location-florida', 'camera-1') or dynamic values from workflow inputs. Tags help organize collected data, filter images in Roboflow, and add metadata for dataset management. Can be an empty list if no tags are needed..
        - `persist_predictions` (*[`boolean`](../kinds/boolean.md)*): If True, model predictions are saved as annotations (pre-labels) in the Roboflow dataset alongside images. This enables predictions to serve as starting points for annotation, allowing reviewers to correct or approve labels rather than creating them from scratch. If False, only images are uploaded without annotations. Enabling this accelerates annotation workflows by providing initial labels..
        - `disable_sink` (*[`boolean`](../kinds/boolean.md)*): If True, the block execution is disabled and no uploads occur. This allows temporarily disabling data collection without removing the block from workflows, useful for testing, debugging, or conditional data collection. When disabled, returns a message indicating the sink was disabled. Default is False (uploads enabled)..
        - `fire_and_forget` (*[`boolean`](../kinds/boolean.md)*): If True, uploads execute asynchronously (fire-and-forget mode), allowing the workflow to continue immediately without waiting for upload completion. This improves workflow performance but prevents error handling. If False, uploads execute synchronously, blocking workflow execution until completion and allowing proper error handling and status reporting. Use async mode (True) for production workflows where speed is prioritized, and sync mode (False) for debugging or when error handling is critical..
        - `labeling_batch_prefix` (*[`string`](../kinds/string.md)*): Prefix used to generate labeling batch names for organizing uploaded images in Roboflow. Combined with the batch recreation frequency and timestamps to create batch names like 'workflows_data_collector_2024_01_15'. Batches help organize collected data for labeling, making it easier to manage and review uploaded images in groups. Can be customized to match your organization scheme..
        - `image_name` (*[`string`](../kinds/string.md)*): Optional custom name for the uploaded image. This is useful when you want to preserve the original filename or use a meaningful identifier (e.g., serial number, timestamp) for the image in the Roboflow dataset. The name should not include file extension. If not provided, a UUID will be generated automatically..
        - `metadata` (*[`*`](../kinds/wildcard.md)*): Optional key-value metadata to attach to uploaded images. Metadata is stored as user_metadata on the image in Roboflow and can be used for filtering and organization. Values can be static strings, numbers, booleans, or references to workflow inputs/steps..

    - output
    
        - `error_status` ([`boolean`](../kinds/boolean.md)): Boolean flag.
        - `message` ([`string`](../kinds/string.md)): String value.



??? tip "Example JSON definition of step `Roboflow Dataset Upload` in version `v2`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/roboflow_dataset_upload@v2",
	    "images": "$inputs.image",
	    "target_project": "my_dataset",
	    "predictions": "$steps.object_detection_model.predictions",
	    "data_percentage": 100,
	    "minutely_usage_limit": 10,
	    "hourly_usage_limit": 10,
	    "daily_usage_limit": 10,
	    "usage_quota_name": "quota-for-data-sampling-1",
	    "max_image_size": [
	        1920,
	        1080
	    ],
	    "compression_level": 95,
	    "registration_tags": [
	        "location-florida",
	        "factory-name",
	        "$inputs.dynamic_tag"
	    ],
	    "persist_predictions": true,
	    "disable_sink": true,
	    "fire_and_forget": "<block_does_not_provide_example>",
	    "labeling_batch_prefix": "my_labeling_batch_name",
	    "labeling_batches_recreation_frequency": "never",
	    "image_name": "serial_12345",
	    "metadata": {
	        "camera_id": "cam_01",
	        "location": "$inputs.location"
	    }
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    




## v1

??? "Class: `RoboflowDatasetUploadBlockV1`  *(there are multiple versions of this block)*"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/sinks/roboflow/dataset_upload/v1.py">inference.core.workflows.core_steps.sinks.roboflow.dataset_upload.v1.RoboflowDatasetUploadBlockV1</a>

    **Warning: This block has multiple versions. Please refer to the specific version for details.**
    You can learn more about how versions work here: [Versioning](/workflows/versioning.md)

    



Upload images and model predictions to a Roboflow dataset for active learning, model improvement, and data collection, with configurable usage quotas, batch organization, image compression, and optional annotation persistence.

## How This Block Works

This block uploads workflow images and predictions to your Roboflow dataset for storage, labeling, and model training. The block:

1. Takes images and optional model predictions (object detection, instance segmentation, keypoint detection, or classification) as input
2. Validates the Roboflow API key is available (required for uploading)
3. Checks usage quotas (minutely, hourly, daily limits) to ensure uploads stay within configured rate limits for active learning strategies
4. Prepares images by resizing if they exceed maximum size (maintaining aspect ratio) and compressing to specified quality level
5. Generates labeling batch names based on the prefix and batch creation frequency (never, daily, weekly, or monthly), organizing uploaded data into batches
6. Optionally persists model predictions as annotations if `persist_predictions` is enabled, allowing predictions to serve as pre-labels for review and correction
7. Attaches registration tags to images for organization and filtering in the Roboflow platform
8. Registers the image (and annotations if enabled) to the specified Roboflow project via the Roboflow API
9. Executes synchronously or asynchronously based on `fire_and_forget` setting, allowing non-blocking uploads for faster workflow execution
10. Returns error status and messages indicating upload success or failure

The block supports active learning workflows by implementing usage quotas that prevent excessive data collection, helping focus on collecting valuable training data within rate limits. Images are organized into labeling batches that can be automatically recreated on a schedule (daily, weekly, monthly), making it easier to manage and review collected data over time. The block can operate in fire-and-forget mode for asynchronous execution, allowing workflows to continue processing without waiting for uploads to complete, or synchronously for debugging and error handling.

## Requirements

**API Key Required**: This block requires a valid Roboflow API key to upload data. The API key must be configured in your environment or workflow configuration. Visit https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key to learn how to retrieve an API key.

## Common Use Cases

- **Active Learning Data Collection**: Collect images and predictions from production environments where models struggle or are uncertain (e.g., low-confidence detections, edge cases), enabling iterative model improvement by gathering challenging examples for retraining
- **Production Data Logging**: Continuously upload production inference data to Roboflow datasets for monitoring, analysis, and future model training, creating a growing dataset from real-world deployments
- **Pre-Labeled Data Collection**: Upload images with model predictions as pre-labels (when `persist_predictions` is enabled), accelerating annotation workflows by providing initial labels that can be reviewed and corrected rather than starting from scratch
- **Stratified Data Sampling**: Use rate limiting and quotas to selectively collect data based on specific criteria (e.g., combine with Rate Limiter or Continue If blocks), ensuring diverse and balanced dataset collection without overwhelming storage or annotation resources
- **Batch-Based Labeling Workflows**: Organize uploaded data into batches with automatic recreation schedules (daily, weekly, monthly), making it easier to manage labeling tasks, track progress, and organize data collection efforts over time
- **Tagged Data Organization**: Attach metadata tags to uploaded images (e.g., location, camera ID, time period, model version), enabling filtering and organization of collected data in Roboflow for better dataset management and analysis

## Connecting to Other Blocks

This block receives data from workflow steps and uploads it to Roboflow:

- **After detection or analysis blocks** (e.g., Object Detection Model, Instance Segmentation Model, Classification Model, Keypoint Detection Model) to upload images along with their predictions, enabling active learning by collecting inference data with model outputs for annotation and retraining
- **After filtering or analytics blocks** (e.g., Detections Filter, Continue If, Overlap Filter) to selectively upload only specific types of data (e.g., low-confidence detections, overlapping objects, specific classes), focusing data collection on valuable edge cases or interesting scenarios
- **After rate limiter blocks** (e.g., Rate Limiter) to throttle upload frequency and stay within usage quotas, ensuring controlled data collection that respects rate limits and prevents excessive storage usage
- **Image inputs or preprocessing blocks** to upload raw images or processed images (e.g., crops, transformed images) without predictions, enabling collection of image data for future labeling or analysis
- **Conditional workflows** using flow control blocks (e.g., Continue If) to upload data only when certain conditions are met (e.g., upload only when detection count exceeds threshold, upload only errors or failures), enabling selective data collection based on workflow state
- **Batch processing workflows** where multiple images or predictions are generated, allowing bulk upload of workflow outputs to Roboflow datasets for organized data collection and management


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/roboflow_dataset_upload@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `target_project` | `str` | Roboflow project identifier where uploaded images and annotations will be saved. Must be a valid project in your Roboflow workspace. The project name can be specified directly or referenced from workflow inputs.. | ✅ |
| `minutely_usage_limit` | `int` | Maximum number of image uploads allowed per minute for this quota. Part of the usage quota system that enforces rate limits for active learning data collection. Uploads exceeding this limit are skipped to prevent excessive data collection. Works together with hourly_usage_limit and daily_usage_limit to provide multi-level rate limiting.. | ❌ |
| `hourly_usage_limit` | `int` | Maximum number of image uploads allowed per hour for this quota. Part of the usage quota system that enforces rate limits for active learning data collection. Uploads exceeding this limit are skipped to prevent excessive data collection. Works together with minutely_usage_limit and daily_usage_limit to provide multi-level rate limiting.. | ❌ |
| `daily_usage_limit` | `int` | Maximum number of image uploads allowed per day for this quota. Part of the usage quota system that enforces rate limits for active learning data collection. Uploads exceeding this limit are skipped to prevent excessive data collection. Works together with minutely_usage_limit and hourly_usage_limit to provide multi-level rate limiting.. | ❌ |
| `usage_quota_name` | `str` | Unique identifier for tracking usage quotas (minutely, hourly, daily limits). Used internally to manage rate limiting across multiple upload operations. Each unique quota name maintains separate counters, allowing different upload strategies or data collection workflows to have independent rate limits.. | ❌ |
| `max_image_size` | `Tuple[int, int]` | Maximum dimensions (width, height) for uploaded images. Images exceeding these dimensions are automatically resized while preserving aspect ratio before uploading. Smaller sizes reduce storage and bandwidth but may lose image quality. Use larger sizes (e.g., (1920, 1080)) for high-resolution data collection, or smaller sizes (e.g., (512, 512)) for efficient storage and faster uploads.. | ❌ |
| `compression_level` | `int` | JPEG compression quality level for uploaded images, ranging from 1 (highest compression, smallest file size, lower quality) to 100 (no compression, largest file size, highest quality). Higher values preserve more image quality but increase storage and bandwidth usage. Typical values range from 70-90 for balanced quality and size. Default of 75 provides good quality with reasonable file sizes.. | ❌ |
| `registration_tags` | `List[str]` | List of tags to attach to uploaded images for organization and filtering in Roboflow. Tags can be static strings (e.g., 'location-florida', 'camera-1') or dynamic values from workflow inputs. Tags help organize collected data, filter images in Roboflow, and add metadata for dataset management. Can be an empty list if no tags are needed.. | ✅ |
| `persist_predictions` | `bool` | If True, model predictions are saved as annotations (pre-labels) in the Roboflow dataset alongside images. This enables predictions to serve as starting points for annotation, allowing reviewers to correct or approve labels rather than creating them from scratch. If False, only images are uploaded without annotations. Enabling this accelerates annotation workflows by providing initial labels.. | ❌ |
| `disable_sink` | `bool` | If True, the block execution is disabled and no uploads occur. This allows temporarily disabling data collection without removing the block from workflows, useful for testing, debugging, or conditional data collection. When disabled, returns a message indicating the sink was disabled. Default is False (uploads enabled).. | ✅ |
| `fire_and_forget` | `bool` | If True, uploads execute asynchronously (fire-and-forget mode), allowing the workflow to continue immediately without waiting for upload completion. This improves workflow performance but prevents error handling. If False, uploads execute synchronously, blocking workflow execution until completion and allowing proper error handling and status reporting. Use async mode (True) for production workflows where speed is prioritized, and sync mode (False) for debugging or when error handling is critical.. | ✅ |
| `labeling_batch_prefix` | `str` | Prefix used to generate labeling batch names for organizing uploaded images in Roboflow. Combined with the batch recreation frequency and timestamps to create batch names like 'workflows_data_collector_2024_01_15'. Batches help organize collected data for labeling, making it easier to manage and review uploaded images in groups. Can be customized to match your organization scheme.. | ✅ |
| `labeling_batches_recreation_frequency` | `str` | Frequency at which new labeling batches are automatically created for uploaded images. Options: 'never' (all images go to the same batch), 'daily' (new batch each day), 'weekly' (new batch each week), 'monthly' (new batch each month). Batch timestamps are appended to the labeling_batch_prefix to create unique batch names. Automatically organizing uploads into time-based batches simplifies dataset management and makes it easier to track and review collected data over time.. | ❌ |
| `image_name` | `str` | Optional custom name for the uploaded image. If provided, this name will be used instead of an auto-generated UUID. This is useful when you want to preserve the original filename or use a meaningful identifier (e.g., serial number, timestamp) for the image in the Roboflow dataset. The name should not include file extension. If not provided, a UUID will be generated automatically.. | ✅ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-cloud-off-outline:{ style="color: #546e7a" } `requires_internet` — air-gapped / offline deployments
:   This block depends on a service that is not reachable from fully offline / air-gapped deployments.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Roboflow Dataset Upload` in version `v1`.

    - inputs: [`YOLO-World Model`](yolo_world_model.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`SAM 3`](sam3.md), [`Dynamic Zone`](dynamic_zone.md), [`Byte Tracker`](byte_tracker.md), [`SORT Tracker`](sort_tracker.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Slack Notification`](slack_notification.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`LMM`](lmm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Image Slicer`](image_slicer.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Velocity`](velocity.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Identify Outliers`](identify_outliers.md), [`Corner Visualization`](corner_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`PLC Reader`](plc_reader.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`VLM As Detector`](vlm_as_detector.md), [`Background Subtraction`](background_subtraction.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Moondream2`](moondream2.md), [`Motion Detection`](motion_detection.md), [`SAM 3`](sam3.md), [`CogVLM`](cog_vlm.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`SIFT`](sift.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Vision OCR`](google_vision_ocr.md), [`SIFT Comparison`](sift_comparison.md), [`Size Measurement`](size_measurement.md), [`CSV Formatter`](csv_formatter.md), [`Contrast Equalization`](contrast_equalization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`EasyOCR`](easy_ocr.md), [`Template Matching`](template_matching.md), [`Google Gemini`](google_gemini.md), [`PLC Writer`](plc_writer.md), [`GLM-OCR`](glmocr.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Stitch Images`](stitch_images.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Circle Visualization`](circle_visualization.md), [`OpenAI`](open_ai.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Image Stack`](image_stack.md), [`Relative Static Crop`](relative_static_crop.md), [`Gaze Detection`](gaze_detection.md), [`Path Deviation`](path_deviation.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`VLM As Detector`](vlm_as_detector.md), [`QR Code Generator`](qr_code_generator.md), [`Dot Visualization`](dot_visualization.md), [`Mask Visualization`](mask_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Blur Visualization`](blur_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Trace Visualization`](trace_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Halo Visualization`](halo_visualization.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Florence-2 Model`](florence2_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Local File Sink`](local_file_sink.md), [`Email Notification`](email_notification.md), [`Detections Transformation`](detections_transformation.md), [`Background Color Visualization`](background_color_visualization.md), [`Image Contours`](image_contours.md), [`Qwen-VL`](qwen_vl.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Florence-2 Model`](florence2_model.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Clip Comparison`](clip_comparison.md), [`MQTT Writer`](mqtt_writer.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Detections Consensus`](detections_consensus.md), [`Time in Zone`](timein_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`OCR Model`](ocr_model.md), [`JSON Parser`](json_parser.md), [`Byte Tracker`](byte_tracker.md), [`Image Threshold`](image_threshold.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Image Blur`](image_blur.md), [`Depth Estimation`](depth_estimation.md), [`Google Gemini`](google_gemini.md), [`Email Notification`](email_notification.md), [`Crop Visualization`](crop_visualization.md), [`Time in Zone`](timein_zone.md), [`Label Visualization`](label_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Track Class Lock`](track_class_lock.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI`](open_ai.md), [`Text Display`](text_display.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Buffer`](buffer.md), [`Image Slicer`](image_slicer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemma API`](google_gemma_api.md), [`Image Preprocessing`](image_preprocessing.md), [`Triangle Visualization`](triangle_visualization.md), [`SAM 3`](sam3.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Identify Changes`](identify_changes.md), [`Halo Visualization`](halo_visualization.md), [`Path Deviation`](path_deviation.md), [`Dimension Collapse`](dimension_collapse.md), [`OpenRouter`](open_router.md), [`Detection Event Log`](detection_event_log.md), [`Detections Combine`](detections_combine.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Byte Tracker`](byte_tracker.md), [`Line Counter`](line_counter.md), [`Overlap Filter`](overlap_filter.md), [`Grid Visualization`](grid_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Detections Stitch`](detections_stitch.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Seg Preview`](seg_preview.md), [`Google Gemma`](google_gemma.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Object Detection Model`](object_detection_model.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Event Writer`](event_writer.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Detections Merge`](detections_merge.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`SIFT Comparison`](sift_comparison.md), [`Detection Offset`](detection_offset.md), [`Keypoint Visualization`](keypoint_visualization.md), [`S3 Sink`](s3_sink.md), [`Camera Focus`](camera_focus.md), [`Color Visualization`](color_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Webhook Sink`](webhook_sink.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Icon Visualization`](icon_visualization.md), [`Detections Filter`](detections_filter.md), [`OpenAI`](open_ai.md), [`Current Time`](current_time.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`SAM2 Video Tracker`](sam2_video_tracker.md)
    - outputs: [`YOLO-World Model`](yolo_world_model.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`SAM 3`](sam3.md), [`Dynamic Zone`](dynamic_zone.md), [`Slack Notification`](slack_notification.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`LMM`](lmm.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Pixelate Visualization`](pixelate_visualization.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Corner Visualization`](corner_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Moondream2`](moondream2.md), [`Motion Detection`](motion_detection.md), [`SAM 3`](sam3.md), [`CogVLM`](cog_vlm.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Anthropic Claude`](anthropic_claude.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Google Vision OCR`](google_vision_ocr.md), [`SIFT Comparison`](sift_comparison.md), [`Size Measurement`](size_measurement.md), [`Contrast Equalization`](contrast_equalization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`PLC Writer`](plc_writer.md), [`Template Matching`](template_matching.md), [`Google Gemini`](google_gemini.md), [`GLM-OCR`](glmocr.md), [`Cache Get`](cache_get.md), [`Circle Visualization`](circle_visualization.md), [`OpenAI`](open_ai.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Image Stack`](image_stack.md), [`Gaze Detection`](gaze_detection.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Path Deviation`](path_deviation.md), [`QR Code Generator`](qr_code_generator.md), [`Dot Visualization`](dot_visualization.md), [`Mask Visualization`](mask_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Pixel Color Count`](pixel_color_count.md), [`Blur Visualization`](blur_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Trace Visualization`](trace_visualization.md), [`Dynamic Crop`](dynamic_crop.md), [`Halo Visualization`](halo_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Florence-2 Model`](florence2_model.md), [`Morphological Transformation`](morphological_transformation.md), [`Local File Sink`](local_file_sink.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Email Notification`](email_notification.md), [`Background Color Visualization`](background_color_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Florence-2 Model`](florence2_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Cache Set`](cache_set.md), [`Clip Comparison`](clip_comparison.md), [`MQTT Writer`](mqtt_writer.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Detections Consensus`](detections_consensus.md), [`Time in Zone`](timein_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`Image Threshold`](image_threshold.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Image Blur`](image_blur.md), [`Depth Estimation`](depth_estimation.md), [`Google Gemini`](google_gemini.md), [`Email Notification`](email_notification.md), [`Crop Visualization`](crop_visualization.md), [`Time in Zone`](timein_zone.md), [`Label Visualization`](label_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI`](open_ai.md), [`Text Display`](text_display.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemma API`](google_gemma_api.md), [`Triangle Visualization`](triangle_visualization.md), [`SAM 3`](sam3.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Distance Measurement`](distance_measurement.md), [`Image Preprocessing`](image_preprocessing.md), [`Halo Visualization`](halo_visualization.md), [`Path Deviation`](path_deviation.md), [`OpenRouter`](open_router.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Line Counter`](line_counter.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Seg Preview`](seg_preview.md), [`Google Gemma`](google_gemma.md), [`Line Counter`](line_counter.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Object Detection Model`](object_detection_model.md), [`Event Writer`](event_writer.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Keypoint Visualization`](keypoint_visualization.md), [`S3 Sink`](s3_sink.md), [`Color Visualization`](color_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Webhook Sink`](webhook_sink.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Icon Visualization`](icon_visualization.md), [`OpenAI`](open_ai.md), [`Current Time`](current_time.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Ellipse Visualization`](ellipse_visualization.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Roboflow Dataset Upload` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `image` (*[`image`](../kinds/image.md)*): Image(s) to upload to the Roboflow dataset. Can be a single image or batch of images from workflow inputs or processing steps. Images are resized if they exceed max_image_size and compressed before uploading. Supports batch processing..
        - `predictions` (*Union[[`object_detection_prediction`](../kinds/object_detection_prediction.md), [`classification_prediction`](../kinds/classification_prediction.md), [`keypoint_detection_prediction`](../kinds/keypoint_detection_prediction.md), [`instance_segmentation_prediction`](../kinds/instance_segmentation_prediction.md)]*): Optional model predictions to upload alongside images. Predictions are saved as annotations (pre-labels) in the Roboflow dataset when persist_predictions is enabled, allowing predictions to serve as starting points for annotation review and correction. Supports object detection, instance segmentation, keypoint detection, and classification predictions. If None, only images are uploaded..
        - `target_project` (*[`roboflow_project`](../kinds/roboflow_project.md)*): Roboflow project identifier where uploaded images and annotations will be saved. Must be a valid project in your Roboflow workspace. The project name can be specified directly or referenced from workflow inputs..
        - `registration_tags` (*Union[[`string`](../kinds/string.md), [`list_of_values`](../kinds/list_of_values.md)]*): List of tags to attach to uploaded images for organization and filtering in Roboflow. Tags can be static strings (e.g., 'location-florida', 'camera-1') or dynamic values from workflow inputs. Tags help organize collected data, filter images in Roboflow, and add metadata for dataset management. Can be an empty list if no tags are needed..
        - `disable_sink` (*[`boolean`](../kinds/boolean.md)*): If True, the block execution is disabled and no uploads occur. This allows temporarily disabling data collection without removing the block from workflows, useful for testing, debugging, or conditional data collection. When disabled, returns a message indicating the sink was disabled. Default is False (uploads enabled)..
        - `fire_and_forget` (*[`boolean`](../kinds/boolean.md)*): If True, uploads execute asynchronously (fire-and-forget mode), allowing the workflow to continue immediately without waiting for upload completion. This improves workflow performance but prevents error handling. If False, uploads execute synchronously, blocking workflow execution until completion and allowing proper error handling and status reporting. Use async mode (True) for production workflows where speed is prioritized, and sync mode (False) for debugging or when error handling is critical..
        - `labeling_batch_prefix` (*[`string`](../kinds/string.md)*): Prefix used to generate labeling batch names for organizing uploaded images in Roboflow. Combined with the batch recreation frequency and timestamps to create batch names like 'workflows_data_collector_2024_01_15'. Batches help organize collected data for labeling, making it easier to manage and review uploaded images in groups. Can be customized to match your organization scheme..
        - `image_name` (*[`string`](../kinds/string.md)*): Optional custom name for the uploaded image. If provided, this name will be used instead of an auto-generated UUID. This is useful when you want to preserve the original filename or use a meaningful identifier (e.g., serial number, timestamp) for the image in the Roboflow dataset. The name should not include file extension. If not provided, a UUID will be generated automatically..

    - output
    
        - `error_status` ([`boolean`](../kinds/boolean.md)): Boolean flag.
        - `message` ([`string`](../kinds/string.md)): String value.



??? tip "Example JSON definition of step `Roboflow Dataset Upload` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/roboflow_dataset_upload@v1",
	    "image": "$inputs.image",
	    "predictions": "$steps.object_detection_model.predictions",
	    "target_project": "my_project",
	    "minutely_usage_limit": 10,
	    "hourly_usage_limit": 10,
	    "daily_usage_limit": 10,
	    "usage_quota_name": "quota-for-data-sampling-1",
	    "max_image_size": [
	        512,
	        512
	    ],
	    "compression_level": 75,
	    "registration_tags": [
	        "location-florida",
	        "factory-name",
	        "$inputs.dynamic_tag"
	    ],
	    "persist_predictions": true,
	    "disable_sink": true,
	    "fire_and_forget": true,
	    "labeling_batch_prefix": "my_labeling_batch_name",
	    "labeling_batches_recreation_frequency": "never",
	    "image_name": "serial_12345"
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    


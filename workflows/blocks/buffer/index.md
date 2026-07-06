
# Buffer



??? "Class: `BufferBlockV1`"

    Source:
    <a target="_blank" href="https://github.com/roboflow/inference/blob/main/inference/core/workflows/core_steps/fusion/buffer/v1.py">inference.core.workflows.core_steps.fusion.buffer.v1.BufferBlockV1</a>
    



Maintain a sliding window buffer of the last N values by storing recent inputs in a FIFO (First-In-First-Out) queue, with newest elements added to the beginning and oldest elements automatically removed when the buffer exceeds the specified length, enabling temporal data collection, frame history tracking, batch processing preparation, and sliding window analysis workflows.

## How This Block Works

This block maintains a rolling buffer that stores the most recent values passed to it, creating a sliding window of data over time. The block:

1. Receives input data of any type (images, detections, values, etc.) and configuration parameters (buffer length and padding option)
2. Maintains an internal buffer that persists across workflow executions:
   - Buffer is initialized as an empty list when the block is first created
   - Buffer state persists for the lifetime of the workflow execution
   - Each buffer block instance maintains its own separate buffer
3. Adds new data to the buffer:
   - Inserts the newest value at the beginning (index 0) of the buffer array
   - Most recent values appear first in the buffer
   - Older values are shifted to later positions in the array
4. Manages buffer size:
   - When buffer length exceeds the specified `length` parameter, removes the oldest elements
   - Keeps only the most recent `length` values
   - Automatically maintains the sliding window size
5. Applies optional padding:
   - If `pad` is True: Fills the buffer with `None` values until it reaches exactly `length` elements
   - Ensures consistent buffer size even when fewer than `length` values have been received
   - If `pad` is False: Buffer size grows from 0 to `length` as values are added, then stays at `length`
6. Returns the buffered array:
   - Outputs a list containing the buffered values in order (newest first)
   - List length equals `length` (if padding enabled) or current buffer size (if padding disabled)
   - Values are ordered from most recent (index 0) to oldest (last index)

The buffer implements a sliding window pattern where new data enters at the front and old data exits at the back when capacity is reached. This creates a temporal history of recent values, useful for operations that need to look back at previous frames, detections, or measurements. The buffer works with any data type, making it flexible for images, detections, numeric values, or other workflow outputs.

## Common Use Cases

- **Frame History Tracking**: Maintain a history of recent video frames for temporal analysis (e.g., track frame sequences, maintain recent image history, collect frames for comparison), enabling temporal frame analysis workflows
- **Detection History**: Buffer recent detections for trend analysis or comparison (e.g., track detection changes over time, compare current vs previous detections, analyze detection patterns), enabling detection history workflows
- **Batch Processing Preparation**: Collect multiple values before processing them together (e.g., batch process recent images, aggregate multiple detections, prepare data for batch operations), enabling batch processing workflows
- **Sliding Window Analysis**: Perform analysis on a rolling window of data (e.g., analyze trends over recent frames, calculate moving averages, detect changes in sequences), enabling sliding window analysis workflows
- **Visualization Sequences**: Maintain recent data for animation or sequence visualization (e.g., create frame sequences, visualize temporal changes, display recent history), enabling temporal visualization workflows
- **Temporal Comparison**: Compare current values with recent historical values (e.g., compare current frame with previous frames, detect changes over time, analyze temporal patterns), enabling temporal comparison workflows

## Connecting to Other Blocks

This block receives data of any type and produces a buffered output array:

- **After any block** that produces values to buffer (e.g., buffer images from image sources, buffer detections from detection models, buffer values from analytics blocks), enabling data buffering workflows
- **Before blocks that process arrays** to provide batched or historical data (e.g., process buffered images, analyze detection arrays, work with value sequences), enabling array processing workflows
- **Before visualization blocks** to display sequences or temporal data (e.g., visualize frame sequences, display detection history, show temporal patterns), enabling temporal visualization workflows
- **Before analysis blocks** that require historical data (e.g., analyze trends over time, compare current vs historical, process temporal sequences), enabling temporal analysis workflows
- **Before aggregation blocks** to provide multiple values for aggregation (e.g., aggregate buffered values, process multiple detections, combine recent data), enabling aggregation workflows
- **In temporal processing pipelines** where maintaining recent history is required (e.g., track changes over time, maintain frame sequences, collect data for temporal analysis), enabling temporal processing workflows

## Requirements

This block works with any data type (images, detections, values, etc.). The buffer maintains state across workflow executions within the same workflow instance. The `length` parameter determines the maximum number of values to keep in the buffer. When `pad` is enabled, the buffer will always return exactly `length` elements (padded with `None` if needed). When `pad` is disabled, the buffer grows from 0 to `length` elements as values are added, then maintains `length` elements by removing oldest values. The buffer persists for the lifetime of the workflow execution and resets when the workflow is restarted.


### Type identifier

Use the following identifier in step `"type"` field: `roboflow_core/buffer@v1`to add the block as
as step in your workflow.

### Properties

| **Name** | **Type** | **Description** | Refs |
|:---------|:---------|:----------------|:-----|
| `name` | `str` | Enter a unique identifier for this step.. | ❌ |
| `length` | `int` | Maximum number of elements to keep in the buffer. When the buffer exceeds this length, the oldest elements are automatically removed. Determines the size of the sliding window. Must be greater than 0. Typical values range from 2-10 for frame sequences, or higher for longer histories.. | ❌ |
| `pad` | `bool` | Enable padding to maintain consistent buffer size. If True, the buffer is padded with `None` values until it reaches exactly `length` elements, ensuring the output always has `length` items even when fewer values have been received. If False, the buffer grows from 0 to `length` as values are added, then maintains `length` by removing oldest values. Use padding when downstream blocks require a fixed-size array.. | ❌ |

The **Refs** column marks possibility to parametrise the property with dynamic values available 
in `workflow` runtime. See *Bindings* for more info.

### :material-shield-half-full:{ style="color: #5e6c75" } Runtime compatibility

:material-alert-circle-outline:{ style="color: #f57c00" } `soft` — runtime `hosted_serverless`, `dedicated_deployment`; execution `remote`; input `video`
:   Block keeps per-video state in process memory (keyed by video_metadata.video_identifier). With remote step execution on stateless or multi-replica HTTP runtimes, successive requests may be served by different worker processes, so the state resets between calls and the output is meaningless for tracking / counting / aggregation. Use local step execution in an InferencePipeline for stable cross-frame results.

:material-alert-circle-outline:{ style="color: #f57c00" } `soft` — input `image`
:   Block depends on temporal context from video or repeated-frame workflows. With a still image/photo, there is no meaningful history to track, compare, aggregate, or visualize, so the block provides little or no benefit.

### Available Connections { data-search-exclude }

??? tip "Compatible Blocks"
    Check what blocks you can connect to `Buffer` in version `v1`.

    - inputs: [`Detections Filter`](detections_filter.md), [`Absolute Static Crop`](absolute_static_crop.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Qwen-VL`](qwen_vl.md), [`Gaze Detection`](gaze_detection.md), [`Cache Get`](cache_get.md), [`Byte Tracker`](byte_tracker.md), [`Triangle Visualization`](triangle_visualization.md), [`EasyOCR`](easy_ocr.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Detection Event Log`](detection_event_log.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Track Class Lock`](track_class_lock.md), [`YOLO-World Model`](yolo_world_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Anthropic Claude`](anthropic_claude.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Stitch Images`](stitch_images.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Continue If`](continue_if.md), [`Dominant Color`](dominant_color.md), [`Google Gemini`](google_gemini.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`GLM-OCR`](glmocr.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`VLM As Detector`](vlm_as_detector.md), [`QR Code Generator`](qr_code_generator.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Line Counter`](line_counter.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Distance Measurement`](distance_measurement.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SmolVLM2`](smol_vlm2.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Detections Transformation`](detections_transformation.md), [`Corner Visualization`](corner_visualization.md), [`Overlap Analysis`](overlap_analysis.md), [`Rate Limiter`](rate_limiter.md), [`Switch Case`](switch_case.md), [`Buffer`](buffer.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Template Matching`](template_matching.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`JSON Parser`](json_parser.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`LMM`](lmm.md), [`Inner Workflow`](inner_workflow.md), [`Velocity`](velocity.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Morphological Transformation`](morphological_transformation.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Dimension Collapse`](dimension_collapse.md), [`Morphological Transformation`](morphological_transformation.md), [`Camera Calibration`](camera_calibration.md), [`Motion Detection`](motion_detection.md), [`Detection Offset`](detection_offset.md), [`Google Vision OCR`](google_vision_ocr.md), [`Clip Comparison`](clip_comparison.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`OpenAI`](open_ai.md), [`Halo Visualization`](halo_visualization.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Object Detection Model`](object_detection_model.md), [`Identify Changes`](identify_changes.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Detections Combine`](detections_combine.md), [`Contrast Equalization`](contrast_equalization.md), [`CSV Formatter`](csv_formatter.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Expression`](expression.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`MQTT Writer`](mqtt_writer.md), [`Cache Set`](cache_set.md), [`Delta Filter`](delta_filter.md), [`PLC Reader`](plc_reader.md), [`Image Threshold`](image_threshold.md), [`Blur Visualization`](blur_visualization.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Google Gemini`](google_gemini.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Background Subtraction`](background_subtraction.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Object Detection Model`](object_detection_model.md), [`Identify Outliers`](identify_outliers.md), [`Path Deviation`](path_deviation.md), [`QR Code Detection`](qr_code_detection.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`SAM 3`](sam3.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Google Gemma API`](google_gemma_api.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Image Stack`](image_stack.md), [`Reference Path Visualization`](reference_path_visualization.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Qwen3-VL`](qwen3_vl.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Crop Visualization`](crop_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Dynamic Zone`](dynamic_zone.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Camera Focus`](camera_focus.md), [`Image Slicer`](image_slicer.md), [`Overlap Filter`](overlap_filter.md), [`Florence-2 Model`](florence2_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Time in Zone`](timein_zone.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Dynamic Crop`](dynamic_crop.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Event Writer`](event_writer.md), [`Perspective Correction`](perspective_correction.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Line Counter`](line_counter.md), [`Polygon Visualization`](polygon_visualization.md), [`Clip Comparison`](clip_comparison.md), [`OCR Model`](ocr_model.md), [`Roboflow Visual Search Classifier`](roboflow_visual_search_classifier.md), [`Circle Visualization`](circle_visualization.md), [`Slack Notification`](slack_notification.md), [`CogVLM`](cog_vlm.md), [`Contrast Enhancement`](contrast_enhancement.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Grid Visualization`](grid_visualization.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Barcode Detection`](barcode_detection.md), [`Byte Tracker`](byte_tracker.md), [`LMM For Classification`](lmm_for_classification.md), [`Florence-2 Model`](florence2_model.md), [`OpenAI`](open_ai.md), [`Time in Zone`](timein_zone.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Local File Sink`](local_file_sink.md), [`SORT Tracker`](sort_tracker.md), [`Image Blur`](image_blur.md), [`Label Visualization`](label_visualization.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Seg Preview`](seg_preview.md), [`Path Deviation`](path_deviation.md), [`Object Detection Model`](object_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Cosine Similarity`](cosine_similarity.md), [`Pixel Color Count`](pixel_color_count.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Preprocessing`](image_preprocessing.md), [`Trace Visualization`](trace_visualization.md), [`Email Notification`](email_notification.md), [`SIFT Comparison`](sift_comparison.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Detections Consensus`](detections_consensus.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Current Time`](current_time.md), [`Dot Visualization`](dot_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Halo Visualization`](halo_visualization.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Property Definition`](property_definition.md), [`Image Contours`](image_contours.md), [`Camera Focus`](camera_focus.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`OpenRouter`](open_router.md), [`Data Aggregator`](data_aggregator.md), [`S3 Sink`](s3_sink.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`VLM As Detector`](vlm_as_detector.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Google Gemma`](google_gemma.md), [`OpenAI`](open_ai.md), [`Moondream2`](moondream2.md), [`Image Slicer`](image_slicer.md), [`Detections Merge`](detections_merge.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Google Gemini`](google_gemini.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Email Notification`](email_notification.md), [`Byte Tracker`](byte_tracker.md), [`SAM 3`](sam3.md), [`Detections Stitch`](detections_stitch.md), [`Text Display`](text_display.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Size Measurement`](size_measurement.md), [`Qwen3.5`](qwen3.5.md), [`Depth Estimation`](depth_estimation.md), [`Icon Visualization`](icon_visualization.md), [`OpenAI`](open_ai.md), [`Time in Zone`](timein_zone.md), [`GeoTag Detection`](geo_tag_detection.md), [`PLC Writer`](plc_writer.md), [`SIFT Comparison`](sift_comparison.md), [`SIFT`](sift.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Background Color Visualization`](background_color_visualization.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Mask Visualization`](mask_visualization.md)
    - outputs: [`SAM 3`](sam3.md), [`Path Deviation`](path_deviation.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Google Gemma API`](google_gemma_api.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Qwen-VL`](qwen_vl.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Triangle Visualization`](triangle_visualization.md), [`Crop Visualization`](crop_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`YOLO-World Model`](yolo_world_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Anthropic Claude`](anthropic_claude.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Webhook Sink`](webhook_sink.md), [`Florence-2 Model`](florence2_model.md), [`Polygon Visualization`](polygon_visualization.md), [`Time in Zone`](timein_zone.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Perspective Correction`](perspective_correction.md), [`Google Gemini`](google_gemini.md), [`Line Counter`](line_counter.md), [`Polygon Visualization`](polygon_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Clip Comparison`](clip_comparison.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Line Counter`](line_counter.md), [`Circle Visualization`](circle_visualization.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Grid Visualization`](grid_visualization.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Time in Zone`](timein_zone.md), [`LMM For Classification`](lmm_for_classification.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Florence-2 Model`](florence2_model.md), [`Corner Visualization`](corner_visualization.md), [`Buffer`](buffer.md), [`Label Visualization`](label_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Seg Preview`](seg_preview.md), [`Path Deviation`](path_deviation.md), [`Object Detection Model`](object_detection_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Trace Visualization`](trace_visualization.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Email Notification`](email_notification.md), [`Detections Consensus`](detections_consensus.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Dot Visualization`](dot_visualization.md), [`Motion Detection`](motion_detection.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Halo Visualization`](halo_visualization.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Mask Visualization`](mask_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`OpenAI`](open_ai.md), [`Halo Visualization`](halo_visualization.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`OpenRouter`](open_router.md), [`Object Detection Model`](object_detection_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Google Gemma`](google_gemma.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`OpenAI`](open_ai.md), [`Google Gemini`](google_gemini.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Email Notification`](email_notification.md), [`SAM 3`](sam3.md), [`Cache Set`](cache_set.md), [`PLC Reader`](plc_reader.md), [`Size Measurement`](size_measurement.md), [`OpenAI`](open_ai.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Time in Zone`](timein_zone.md), [`Google Gemini`](google_gemini.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`VLM As Classifier`](vlm_as_classifier.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Buffer` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `data` (*Union[[`list_of_values`](../kinds/list_of_values.md), [`*`](../kinds/wildcard.md), [`image`](../kinds/image.md)]*): Input data of any type to add to the buffer. Can be images, detections, values, or any other workflow output. Newest values are added to the beginning of the buffer array. The buffer maintains a sliding window of the most recent values..

    - output
    
        - `output` ([`list_of_values`](../kinds/list_of_values.md)): List of values of any type.



??? tip "Example JSON definition of step `Buffer` in version `v1`"

    ```json
    {
	    "name": "<your_step_name_here>",
	    "type": "roboflow_core/buffer@v1",
	    "data": "$steps.visualization",
	    "length": 5,
	    "pad": true
	}
    ```

<style>
/* hide edit button for generated pages */
article > a.md-content__button.md-icon:first-child {
    display: none;
}
</style>    



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

    - inputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`VLM As Detector`](vlm_as_detector.md), [`Anthropic Claude`](anthropic_claude.md), [`Image Threshold`](image_threshold.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen3.5`](qwen3.5.md), [`OpenAI`](open_ai.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Qwen-VL`](qwen_vl.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Slack Notification`](slack_notification.md), [`Detections Combine`](detections_combine.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Cosine Similarity`](cosine_similarity.md), [`Florence-2 Model`](florence2_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Template Matching`](template_matching.md), [`Dynamic Zone`](dynamic_zone.md), [`Halo Visualization`](halo_visualization.md), [`Track Class Lock`](track_class_lock.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Webhook Sink`](webhook_sink.md), [`SAM 3`](sam3.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`SAM 3`](sam3.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`QR Code Generator`](qr_code_generator.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Google Gemini`](google_gemini.md), [`YOLO-World Model`](yolo_world_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Inner Workflow`](inner_workflow.md), [`Clip Comparison`](clip_comparison.md), [`Byte Tracker`](byte_tracker.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Halo Visualization`](halo_visualization.md), [`Label Visualization`](label_visualization.md), [`SIFT`](sift.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Detection Offset`](detection_offset.md), [`Florence-2 Model`](florence2_model.md), [`Path Deviation`](path_deviation.md), [`Delta Filter`](delta_filter.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Morphological Transformation`](morphological_transformation.md), [`Text Display`](text_display.md), [`MQTT Writer`](mqtt_writer.md), [`PLC Writer`](plc_writer.md), [`Gaze Detection`](gaze_detection.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Google Gemini`](google_gemini.md), [`Google Gemma API`](google_gemma_api.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`PLC Reader`](plc_reader.md), [`Polygon Visualization`](polygon_visualization.md), [`SAM2 Video Tracker`](sam2_video_tracker.md), [`Moondream2`](moondream2.md), [`Expression`](expression.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`Google Gemini`](google_gemini.md), [`Overlap Analysis`](overlap_analysis.md), [`EasyOCR`](easy_ocr.md), [`Motion Detection`](motion_detection.md), [`Time in Zone`](timein_zone.md), [`Time in Zone`](timein_zone.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Image Slicer`](image_slicer.md), [`Polygon Visualization`](polygon_visualization.md), [`Switch Case`](switch_case.md), [`Background Subtraction`](background_subtraction.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Data Aggregator`](data_aggregator.md), [`Image Preprocessing`](image_preprocessing.md), [`Blur Visualization`](blur_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Local File Sink`](local_file_sink.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Rate Limiter`](rate_limiter.md), [`Byte Tracker`](byte_tracker.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`Image Slicer`](image_slicer.md), [`Identify Changes`](identify_changes.md), [`Email Notification`](email_notification.md), [`Line Counter`](line_counter.md), [`Reference Path Visualization`](reference_path_visualization.md), [`S3 Sink`](s3_sink.md), [`LMM`](lmm.md), [`Distance Measurement`](distance_measurement.md), [`Cache Set`](cache_set.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Size Measurement`](size_measurement.md), [`Detections Consensus`](detections_consensus.md), [`Trace Visualization`](trace_visualization.md), [`SAM 3`](sam3.md), [`Circle Visualization`](circle_visualization.md), [`Time in Zone`](timein_zone.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Event Writer`](event_writer.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Path Deviation`](path_deviation.md), [`Contrast Equalization`](contrast_equalization.md), [`Image Blur`](image_blur.md), [`Google Vision OCR`](google_vision_ocr.md), [`Byte Tracker`](byte_tracker.md), [`Morphological Transformation`](morphological_transformation.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Qwen3-VL`](qwen3_vl.md), [`Barcode Detection`](barcode_detection.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`SmolVLM2`](smol_vlm2.md), [`Email Notification`](email_notification.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`OpenRouter`](open_router.md), [`CSV Formatter`](csv_formatter.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`Detections Merge`](detections_merge.md), [`OpenAI`](open_ai.md), [`Grid Visualization`](grid_visualization.md), [`Mask Visualization`](mask_visualization.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Detections Transformation`](detections_transformation.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`LMM For Classification`](lmm_for_classification.md), [`Anthropic Claude`](anthropic_claude.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Dynamic Crop`](dynamic_crop.md), [`Anthropic Claude`](anthropic_claude.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Velocity`](velocity.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Property Definition`](property_definition.md), [`Color Visualization`](color_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Object Detection Model`](object_detection_model.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Cache Get`](cache_get.md), [`Seg Preview`](seg_preview.md), [`Clip Comparison`](clip_comparison.md), [`Overlap Filter`](overlap_filter.md), [`Camera Calibration`](camera_calibration.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Dominant Color`](dominant_color.md), [`Detections Filter`](detections_filter.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Background Color Visualization`](background_color_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`CogVLM`](cog_vlm.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Corner Visualization`](corner_visualization.md), [`SORT Tracker`](sort_tracker.md), [`OpenAI`](open_ai.md), [`Object Detection Model`](object_detection_model.md), [`Identify Outliers`](identify_outliers.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Depth Estimation`](depth_estimation.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Classification Label Visualization`](classification_label_visualization.md), [`SIFT Comparison`](sift_comparison.md), [`Image Stack`](image_stack.md), [`Line Counter`](line_counter.md), [`Current Time`](current_time.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Google Gemma`](google_gemma.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Stitch Images`](stitch_images.md), [`Keypoint Visualization`](keypoint_visualization.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`Icon Visualization`](icon_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Relative Static Crop`](relative_static_crop.md), [`Detections Stitch`](detections_stitch.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`OCR Model`](ocr_model.md), [`Perspective Correction`](perspective_correction.md), [`Buffer`](buffer.md), [`Crop Visualization`](crop_visualization.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Camera Focus`](camera_focus.md), [`JSON Parser`](json_parser.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`GLM-OCR`](glmocr.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Pixel Color Count`](pixel_color_count.md), [`Dimension Collapse`](dimension_collapse.md), [`QR Code Detection`](qr_code_detection.md), [`Detection Event Log`](detection_event_log.md), [`Continue If`](continue_if.md), [`Image Contours`](image_contours.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md)
    - outputs: [`Keypoint Detection Model`](keypoint_detection_model.md), [`VLM As Detector`](vlm_as_detector.md), [`Anthropic Claude`](anthropic_claude.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Triangle Visualization`](triangle_visualization.md), [`Qwen-VL`](qwen_vl.md), [`Email Notification`](email_notification.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`OpenRouter`](open_router.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`OpenAI`](open_ai.md), [`Florence-2 Model`](florence2_model.md), [`Grid Visualization`](grid_visualization.md), [`VLM As Detector`](vlm_as_detector.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Mask Visualization`](mask_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`OpenAI`](open_ai.md), [`LMM For Classification`](lmm_for_classification.md), [`Halo Visualization`](halo_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Webhook Sink`](webhook_sink.md), [`SAM 3`](sam3.md), [`Anthropic Claude`](anthropic_claude.md), [`SAM 3`](sam3.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Color Visualization`](color_visualization.md), [`YOLO-World Model`](yolo_world_model.md), [`Google Gemini`](google_gemini.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Halo Visualization`](halo_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Clip Comparison`](clip_comparison.md), [`Label Visualization`](label_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Path Deviation`](path_deviation.md), [`Florence-2 Model`](florence2_model.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Seg Preview`](seg_preview.md), [`Clip Comparison`](clip_comparison.md), [`Google Gemini`](google_gemini.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Google Gemma API`](google_gemma_api.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`VLM As Classifier`](vlm_as_classifier.md), [`PLC Reader`](plc_reader.md), [`Polygon Visualization`](polygon_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Google Gemini`](google_gemini.md), [`Corner Visualization`](corner_visualization.md), [`Motion Detection`](motion_detection.md), [`OpenAI`](open_ai.md), [`Object Detection Model`](object_detection_model.md), [`Time in Zone`](timein_zone.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Time in Zone`](timein_zone.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Polygon Visualization`](polygon_visualization.md), [`Line Counter`](line_counter.md), [`Google Gemma`](google_gemma.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Dot Visualization`](dot_visualization.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Buffer`](buffer.md), [`Perspective Correction`](perspective_correction.md), [`Crop Visualization`](crop_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Line Counter`](line_counter.md), [`Email Notification`](email_notification.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Cache Set`](cache_set.md), [`Detections Consensus`](detections_consensus.md), [`Size Measurement`](size_measurement.md), [`Trace Visualization`](trace_visualization.md), [`SAM 3`](sam3.md), [`Circle Visualization`](circle_visualization.md), [`Time in Zone`](timein_zone.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Path Deviation`](path_deviation.md)

    
### Input and Output Bindings

The available connections depend on its binding kinds. Check what binding kinds 
`Buffer` in version `v1`  has.

???+ tip "Bindings"

    - input
    
        - `data` (*Union[[`*`](../kinds/wildcard.md), [`list_of_values`](../kinds/list_of_values.md), [`image`](../kinds/image.md)]*): Input data of any type to add to the buffer. Can be images, detections, values, or any other workflow output. Newest values are added to the beginning of the buffer array. The buffer maintains a sliding window of the most recent values..

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



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

    - inputs: [`YOLO-World Model`](yolo_world_model.md), [`Microsoft SQL Server Sink`](microsoft_sql_server_sink.md), [`SAM 3`](sam3.md), [`Dynamic Zone`](dynamic_zone.md), [`Byte Tracker`](byte_tracker.md), [`SORT Tracker`](sort_tracker.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Slack Notification`](slack_notification.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`LMM`](lmm.md), [`Qwen3-VL`](qwen3_vl.md), [`Cosine Similarity`](cosine_similarity.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Image Slicer`](image_slicer.md), [`Per-Class Confidence Filter`](per_class_confidence_filter.md), [`Delta Filter`](delta_filter.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Pixelate Visualization`](pixelate_visualization.md), [`Camera Focus`](camera_focus.md), [`OpenAI`](open_ai.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Object Detection Model`](object_detection_model.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Property Definition`](property_definition.md), [`Velocity`](velocity.md), [`Roboflow Custom Metadata`](roboflow_custom_metadata.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Identify Outliers`](identify_outliers.md), [`Corner Visualization`](corner_visualization.md), [`Trace Visualization`](trace_visualization.md), [`OPC UA Writer Sink`](opcua_writer_sink.md), [`LMM For Classification`](lmm_for_classification.md), [`Morphological Transformation`](morphological_transformation.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`PLC Reader`](plc_reader.md), [`PTZ Tracking (ONVIF)`](ptz_tracking(onvif).md), [`VLM As Detector`](vlm_as_detector.md), [`Background Subtraction`](background_subtraction.md), [`OpenAI-Compatible LLM`](open_ai_compatible_llm.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`Stability AI Outpainting`](stability_ai_outpainting.md), [`Moondream2`](moondream2.md), [`Motion Detection`](motion_detection.md), [`SAM 3`](sam3.md), [`CogVLM`](cog_vlm.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Stability AI Inpainting`](stability_ai_inpainting.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`QR Code Detection`](qr_code_detection.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`SIFT`](sift.md), [`Google Vision OCR`](google_vision_ocr.md), [`Stability AI Image Generation`](stability_ai_image_generation.md), [`SIFT Comparison`](sift_comparison.md), [`Size Measurement`](size_measurement.md), [`CSV Formatter`](csv_formatter.md), [`SmolVLM2`](smol_vlm2.md), [`Contrast Equalization`](contrast_equalization.md), [`Qwen3.5`](qwen3.5.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`EasyOCR`](easy_ocr.md), [`Google Gemini`](google_gemini.md), [`Template Matching`](template_matching.md), [`PLC Writer`](plc_writer.md), [`GLM-OCR`](glmocr.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Stitch Images`](stitch_images.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Cache Get`](cache_get.md), [`Circle Visualization`](circle_visualization.md), [`OpenAI`](open_ai.md), [`Barcode Detection`](barcode_detection.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Roboflow Visual Search`](roboflow_visual_search.md), [`Segment Anything 2 Model`](segment_anything2_model.md), [`Continue If`](continue_if.md), [`OC-SORT Tracker`](ocsort_tracker.md), [`Image Stack`](image_stack.md), [`Relative Static Crop`](relative_static_crop.md), [`Gaze Detection`](gaze_detection.md), [`Path Deviation`](path_deviation.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`VLM As Detector`](vlm_as_detector.md), [`Qwen2.5-VL`](qwen2.5_vl.md), [`QR Code Generator`](qr_code_generator.md), [`Dot Visualization`](dot_visualization.md), [`Mask Visualization`](mask_visualization.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Pixel Color Count`](pixel_color_count.md), [`Perspective Correction`](perspective_correction.md), [`Blur Visualization`](blur_visualization.md), [`Bounding Rectangle`](bounding_rectangle.md), [`Dynamic Crop`](dynamic_crop.md), [`Overlap Analysis`](overlap_analysis.md), [`Dominant Color`](dominant_color.md), [`Halo Visualization`](halo_visualization.md), [`Absolute Static Crop`](absolute_static_crop.md), [`Florence-2 Model`](florence2_model.md), [`Single-Label Classification Model`](single_label_classification_model.md), [`SIFT Comparison`](sift_comparison.md), [`Detections Stabilizer`](detections_stabilizer.md), [`Rate Limiter`](rate_limiter.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Local File Sink`](local_file_sink.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Email Notification`](email_notification.md), [`Morphological Transformation`](morphological_transformation.md), [`Detections Transformation`](detections_transformation.md), [`Background Color Visualization`](background_color_visualization.md), [`Image Contours`](image_contours.md), [`Environment Secrets Store`](environment_secrets_store.md), [`Qwen-VL`](qwen_vl.md), [`PLC ModbusTCP`](plc_modbus_tcp.md), [`Florence-2 Model`](florence2_model.md), [`ByteTrack Tracker`](byte_track_tracker.md), [`Clip Comparison`](clip_comparison.md), [`Cache Set`](cache_set.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`MQTT Writer`](mqtt_writer.md), [`CLIP Embedding Model`](clip_embedding_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Mask Edge Snap`](mask_edge_snap.md), [`Detections Consensus`](detections_consensus.md), [`Time in Zone`](timein_zone.md), [`Inner Workflow`](inner_workflow.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`OCR Model`](ocr_model.md), [`JSON Parser`](json_parser.md), [`Byte Tracker`](byte_tracker.md), [`Image Threshold`](image_threshold.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Image Blur`](image_blur.md), [`Semantic Segmentation Model`](semantic_segmentation_model.md), [`Depth Estimation`](depth_estimation.md), [`Google Gemini`](google_gemini.md), [`Email Notification`](email_notification.md), [`Crop Visualization`](crop_visualization.md), [`Time in Zone`](timein_zone.md), [`Label Visualization`](label_visualization.md), [`Camera Calibration`](camera_calibration.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Track Class Lock`](track_class_lock.md), [`OpenAI`](open_ai.md), [`Text Display`](text_display.md), [`Image Convert Grayscale`](image_convert_grayscale.md), [`Buffer`](buffer.md), [`Image Slicer`](image_slicer.md), [`Distance Measurement`](distance_measurement.md), [`Stitch OCR Detections`](stitch_ocr_detections.md), [`Image Preprocessing`](image_preprocessing.md), [`SAM 3`](sam3.md), [`Triangle Visualization`](triangle_visualization.md), [`BoT-SORT Tracker`](bo_tsort_tracker.md), [`Identify Changes`](identify_changes.md), [`Google Gemma API`](google_gemma_api.md), [`First Non Empty Or Default`](first_non_empty_or_default.md), [`Halo Visualization`](halo_visualization.md), [`Path Deviation`](path_deviation.md), [`Switch Case`](switch_case.md), [`Dimension Collapse`](dimension_collapse.md), [`OpenRouter`](open_router.md), [`Expression`](expression.md), [`Detection Event Log`](detection_event_log.md), [`Detections Combine`](detections_combine.md), [`Qwen3.5-VL`](qwen3.5_vl.md), [`Byte Tracker`](byte_tracker.md), [`Line Counter`](line_counter.md), [`Overlap Filter`](overlap_filter.md), [`Grid Visualization`](grid_visualization.md), [`Contrast Enhancement`](contrast_enhancement.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Clip Comparison`](clip_comparison.md), [`Heatmap Visualization`](heatmap_visualization.md), [`Detections Stitch`](detections_stitch.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Seg Preview`](seg_preview.md), [`Google Gemma`](google_gemma.md), [`Line Counter`](line_counter.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Twilio SMS Notification`](twilio_sms_notification.md), [`Object Detection Model`](object_detection_model.md), [`SAM 3 Interactive`](sam3_interactive.md), [`Event Writer`](event_writer.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Detections Merge`](detections_merge.md), [`Multi-Label Classification Model`](multi_label_classification_model.md), [`Data Aggregator`](data_aggregator.md), [`Detection Offset`](detection_offset.md), [`Keypoint Visualization`](keypoint_visualization.md), [`S3 Sink`](s3_sink.md), [`Camera Focus`](camera_focus.md), [`Color Visualization`](color_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Perception Encoder Embedding Model`](perception_encoder_embedding_model.md), [`Webhook Sink`](webhook_sink.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Model Monitoring Inference Aggregator`](model_monitoring_inference_aggregator.md), [`Model Comparison Visualization`](model_comparison_visualization.md), [`Mask Area Measurement`](mask_area_measurement.md), [`Detections Filter`](detections_filter.md), [`Current Time`](current_time.md), [`Icon Visualization`](icon_visualization.md), [`OpenAI`](open_ai.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Roboflow Vision Events`](roboflow_vision_events.md), [`SAM2 Video Tracker`](sam2_video_tracker.md)
    - outputs: [`YOLO-World Model`](yolo_world_model.md), [`SAM 3`](sam3.md), [`Florence-2 Model`](florence2_model.md), [`Qwen 3.6 API`](qwen3.6_api.md), [`Detections Classes Replacement`](detections_classes_replacement.md), [`Clip Comparison`](clip_comparison.md), [`Cache Set`](cache_set.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Qwen 3.5 API`](qwen3.5_api.md), [`Detections Consensus`](detections_consensus.md), [`Time in Zone`](timein_zone.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`Line Counter Visualization`](line_counter_visualization.md), [`Polygon Visualization`](polygon_visualization.md), [`Roboflow Dataset Upload`](roboflow_dataset_upload.md), [`PLC EthernetIP`](plc_ethernet_ip.md), [`Google Gemini`](google_gemini.md), [`OpenAI`](open_ai.md), [`Email Notification`](email_notification.md), [`Crop Visualization`](crop_visualization.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Object Detection Model`](object_detection_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Time in Zone`](timein_zone.md), [`Label Visualization`](label_visualization.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Anthropic Claude`](anthropic_claude.md), [`Twilio SMS/MMS Notification`](twilio_smsmms_notification.md), [`Buffer`](buffer.md), [`Corner Visualization`](corner_visualization.md), [`LMM For Classification`](lmm_for_classification.md), [`Google Gemma API`](google_gemma_api.md), [`Triangle Visualization`](triangle_visualization.md), [`SAM 3`](sam3.md), [`PLC Reader`](plc_reader.md), [`VLM As Detector`](vlm_as_detector.md), [`Halo Visualization`](halo_visualization.md), [`Path Deviation`](path_deviation.md), [`Motion Detection`](motion_detection.md), [`OpenRouter`](open_router.md), [`SAM 3`](sam3.md), [`Classification Label Visualization`](classification_label_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Llama 3.2 Vision`](llama3.2_vision.md), [`Detections List Roll-Up`](detections_list_roll_up.md), [`Line Counter`](line_counter.md), [`Size Measurement`](size_measurement.md), [`Grid Visualization`](grid_visualization.md), [`Reference Path Visualization`](reference_path_visualization.md), [`Anthropic Claude`](anthropic_claude.md), [`Polygon Zone Visualization`](polygon_zone_visualization.md), [`Google Gemini`](google_gemini.md), [`Clip Comparison`](clip_comparison.md), [`Bounding Box Visualization`](bounding_box_visualization.md), [`Seg Preview`](seg_preview.md), [`VLM As Classifier`](vlm_as_classifier.md), [`VLM As Classifier`](vlm_as_classifier.md), [`Google Gemma`](google_gemma.md), [`Line Counter`](line_counter.md), [`Circle Visualization`](circle_visualization.md), [`OpenAI`](open_ai.md), [`Google Gemini`](google_gemini.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Object Detection Model`](object_detection_model.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Path Deviation`](path_deviation.md), [`SAM3 Video Tracker`](sam3_video_tracker.md), [`VLM As Detector`](vlm_as_detector.md), [`MoonshotAI Kimi`](moonshot_ai_kimi.md), [`Mask Visualization`](mask_visualization.md), [`Dot Visualization`](dot_visualization.md), [`Keypoint Visualization`](keypoint_visualization.md), [`Perspective Correction`](perspective_correction.md), [`Color Visualization`](color_visualization.md), [`Trace Visualization`](trace_visualization.md), [`Halo Visualization`](halo_visualization.md), [`Object Detection Model`](object_detection_model.md), [`Florence-2 Model`](florence2_model.md), [`Instance Segmentation Model`](instance_segmentation_model.md), [`Webhook Sink`](webhook_sink.md), [`Polygon Visualization`](polygon_visualization.md), [`Keypoint Detection Model`](keypoint_detection_model.md), [`Roboflow Asset Library Attributes`](roboflow_asset_library_attributes.md), [`Email Notification`](email_notification.md), [`OpenAI`](open_ai.md), [`Ellipse Visualization`](ellipse_visualization.md), [`Qwen-VL`](qwen_vl.md)

    
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

